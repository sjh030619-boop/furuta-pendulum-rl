"""4단계 3: 멈춤 규칙 + 행동 변화 벌점으로 SAC 학습.

    python s2r/stage4_train.py --du 1.0 --seed 0 [--rule 2x15] [--budget 600] [--every 30]
    python s2r/stage4_train.py --plan --dus 0 1 3        # 변형마다 시드 3개 (3개씩 동시)
    python s2r/stage4_train.py --report

설정은 1단계 SAC 와 같음(stage1.SAC_KW, 환경 8개). 다른 점:
- 보상에 -du*(u-u_prev)^2 (S2REnv du_pen). du=0 이면 1단계와 같은 보상.
- 학습 30 s 마다 빠른 평가 32칸(조건 8개 x 시작 4개: θ0=0 의 α0 -30°, +5°, θ0=1 rad 의 α0 -5°, +30°,
  엄격 기준), 멈춤 규칙을 만족하면 그 자리에서 멈춤(최대 budget). 규칙 "AxB": B/32 이상을 A번 연속.
  공칭 16칸만 보면 너무 일찍 16/16 이 되어 128칸 74~117 인 모델에서 멈췄음(stage4_ckpt.py 결과).
멈춘 모델로 128칸 평가와 시뮬 직립 전류(stage4_chatter)를 잼.
"""
import time

T0 = time.perf_counter()

import os
import sys
import json
import glob
import math
import argparse
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "rl"))
OUT = os.path.join(HERE, "runs_stage4")


def run_one(du, seed, rule, budget, every, out):
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from s2r_env import make_env, S2REnv, NOMINAL
    from evaluate import evaluate, run_episode, CONDITIONS, STARTS
    from stage1 import SAC_KW, NET
    from stage4_chatter import hold_metrics

    torch.set_num_threads(1)
    need_n, need_q = (int(x) for x in rule.split("x"))
    tag = "du%g_s%d" % (du, seed)
    venv = DummyVecEnv([make_env(randomize=True, du_pen=du) for _ in range(8)])
    venv.seed(seed)
    model = SAC("MlpPolicy", venv, gamma=0.99, seed=seed, device="cpu", verbose=0,
                policy_kwargs=dict(net_arch=NET), **SAC_KW)

    def pol(o):
        return float(model.predict(o, deterministic=True)[0][0])

    quick_starts = [STARTS[i] for i in (0, 5, 10, 15)]
    qenvs = {}
    for name, over in CONDITIONS.items():
        qenvs[name] = S2REnv(params=dict(NOMINAL, **over), episode_sec=12.0, init="hang")
        qenvs[name].reset(seed=0)

    def quick():
        sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.policy.actor.state_dict().items()}
        W1, b1, W2, b2 = sd["latent_pi.0.weight"], sd["latent_pi.0.bias"], sd["latent_pi.2.weight"], sd["latent_pi.2.bias"]
        Wm, bm = sd["mu.weight"], sd["mu.bias"]

        def npol(o):
            h = np.maximum(0.0, W1 @ o.astype(np.float64) + b1)
            h = np.maximum(0.0, W2 @ h + b2)
            return float(np.tanh((Wm @ h + bm)[0]))
        return sum(run_episode(e, npol, th, al)["strict"] for e in qenvs.values() for th, al in quick_starts)

    class CB(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.train, self.mark, self.next = 0.0, None, every
            self.points, self.stopped = [], None

        def _snap(self):
            self.train += time.perf_counter() - self.mark
            te = time.perf_counter()
            q32 = quick()
            self.points.append(dict(t=round(self.train, 1), steps=int(self.num_timesteps), strict=q32,
                                    eval_s=round(time.perf_counter() - te, 1)))
            print("%s t=%6.1f s  steps=%7d  빠른 평가 %2d/32 (%.1f s)" % (
                tag, self.train, self.num_timesteps, q32, time.perf_counter() - te), flush=True)
            self.next = self.train + every
            self.mark = time.perf_counter()
            q = [p["strict"] for p in self.points]
            return len(q) >= need_n and all(v >= need_q for v in q[-need_n:])

        def _on_training_start(self):
            self.mark = time.perf_counter()

        def _on_step(self):
            e = self.train + (time.perf_counter() - self.mark)
            if e >= self.next or e >= budget:
                ok = self._snap()
                if ok:
                    self.stopped = "rule"
                    return False
                if self.train >= budget:
                    self.stopped = "budget"
                    return False
            return True

    cb = CB()
    model.learn(total_timesteps=10 ** 10, callback=cb, progress_bar=False)
    venv.close()
    model.save(os.path.join(out, "sac_%s" % tag))
    full = evaluate(pol, verbose=False)
    env = S2REnv(params=dict(NOMINAL), episode_sec=12.0, init="hang")
    env.reset(seed=0)
    starts = [math.radians(a) for a in (30, -30, 15, -15, 5, -5, 2, -2)]
    iq, dU, rms = hold_metrics(pol, env, starts)
    rec = dict(du=du, seed=seed, rule=rule, budget=budget, every=every, stopped=cb.stopped,
               t_stop=cb.points[-1]["t"] if cb.points else None, steps=int(model.num_timesteps),
               full_strict=full["total_strict"], full_loose=full["total_loose"],
               per={k: v["strict"] for k, v in full.items() if isinstance(v, dict)},
               eval_s=round(sum(p["eval_s"] for p in cb.points), 1), hold_iq=iq, hold_du=dU, hold_rms_deg=rms, points=cb.points, setup_s=round(time.perf_counter() - T0, 1))
    with open(os.path.join(out, "train_%s.json" % tag), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print("%s 끝: %s %.0f s 에 멈춤, 128칸 엄격 %d, 직립 |iq| %.3f A, |Δu| %.3f" % (
        tag, cb.stopped, rec["t_stop"] or -1, full["total_strict"], iq, dU), flush=True)


def plan(dus, seeds, rule, budget, every, out):
    py = sys.executable
    me = os.path.abspath(__file__)
    logdir = os.path.join(out, "logs")
    os.makedirs(logdir, exist_ok=True)
    jobs = [(d, s) for d in dus for s in seeds]
    t0 = time.time()
    for i in range(0, len(jobs), 3):
        procs = []
        for d, s in jobs[i:i + 3]:
            cmd = [py, "-u", me, "--du", str(d), "--seed", str(s), "--rule", rule, "--budget", str(budget),
                   "--every", str(every), "--out", out]
            procs.append(subprocess.Popen(cmd, stdout=open(os.path.join(logdir, "du%g_s%d.log" % (d, s)), "w"),
                                          stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()
        print("묶음 %d 끝 (누적 %.1f 분)" % (i // 3 + 1, (time.time() - t0) / 60), flush=True)
    report(out)


def report(out):
    rows = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(out, "train_du*.json")))]
    print("%-6s %4s %8s %8s %9s %10s %9s" % ("du", "시드", "멈춤", "시각 s", "128칸", "직립|iq|", "|Δu|"))
    for r in rows:
        print("%-6g %4d %8s %8.0f %6d/128 %8.3f A %9.3f" % (r["du"], r["seed"], r["stopped"], r["t_stop"],
                                                           r["full_strict"], r["hold_iq"], r["hold_du"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--du", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", default="2x30")
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--every", type=float, default=30.0)
    ap.add_argument("--dus", type=float, nargs="+", default=[0.0, 1.0, 3.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.report:
        report(a.out)
    elif a.plan:
        plan(a.dus, a.seeds, a.rule, a.budget, a.every, a.out)
    else:
        run_one(a.du, a.seed, a.rule, a.budget, a.every, a.out)


if __name__ == "__main__":
    main()
