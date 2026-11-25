"""앱의 '따라 하기' 아홉 칸 정책을 rev11 보정 환경에서 SAC 로 학습합니다.

    python s2r/train_app_rules.py --plan                 # 아홉 칸 전부 (3개씩 동시)
    python s2r/train_app_rules.py --s 1 --e 2            # 한 칸
    python s2r/train_app_rules.py --report

학습지의 두 칸(부드럽게, 힘 아끼기)이 각각 없음/약하게/강하게이고, 보상의 가중치만 다릅니다.
    r = 1 - 0.3 phi^2 - w_smooth ald^2 - 0.002 thd^2 - w_effort u^2 + 0.5 near slow   (종료 벌점 -200)
    w_smooth = [0, 0.002, 0.008],  w_effort = [0, 0.02, 0.10]      (rl/train_rules.py, 앱 reward.js 와 같은 값)
가운데 칸(약하게, 약하게)이 1단계에서 쓴 보상과 같습니다.

설정은 1단계 SAC 그대로(환경 8개, 학습률 1e-3). 30 s 마다 고정 시험 32칸과 그 규칙의 평균 점수를 재고,
30/32 이상이 2번 연속이면 멈춥니다(최대 budget). 멈춘 모델이 아니라 **시험 점수가 가장 좋았던 순간**의 모델을 저장합니다.
학습 곡선(curve)은 앱이 '배우면서 받은 점수'로 그대로 보여 줍니다.
"""
import time

T0 = time.perf_counter()

import os
import sys
import json
import glob
import argparse
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "rl"))
OUT = os.path.join(HERE, "runs_app_rules")

W_SMOOTH = [0.0, 0.002, 0.008]
W_EFFORT = [0.0, 0.02, 0.10]
WORD = ["없음", "약하게", "강하게"]
KEYS = ["latent_pi.0.weight", "latent_pi.0.bias", "latent_pi.2.weight", "latent_pi.2.bias", "mu.weight", "mu.bias"]


def run_one(s, e, seed, budget, every, out):
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from s2r_env import make_env, S2REnv, NOMINAL
    from evaluate import evaluate, run_episode, CONDITIONS, STARTS
    from stage1 import SAC_KW, NET

    torch.set_num_threads(1)
    ws, we = W_SMOOTH[s], W_EFFORT[e]
    tag = "sac_s%de%d" % (s, e)
    venv = DummyVecEnv([make_env(randomize=True, w_smooth=ws, w_effort=we) for _ in range(8)])
    venv.seed(seed)
    model = SAC("MlpPolicy", venv, gamma=0.99, seed=seed, device="cpu", verbose=0,
                policy_kwargs=dict(net_arch=NET), **SAC_KW)
    quick_starts = [STARTS[i] for i in (0, 5, 10, 15)]
    qenvs = []
    for over in CONDITIONS.values():
        env = S2REnv(params=dict(NOMINAL, **over), episode_sec=12.0, init="hang", w_smooth=ws, w_effort=we)
        env.reset(seed=0)
        qenvs.append(env)

    def weights():
        sd = model.policy.actor.state_dict()
        return {k: sd[k].detach().cpu().numpy().astype(np.float64).copy() for k in KEYS}

    def np_pol(w):
        W1, b1, W2, b2, Wm, bm = (w[k] for k in KEYS)

        def pol(o):
            h = np.maximum(0.0, W1 @ o.astype(np.float64) + b1)
            h = np.maximum(0.0, W2 @ h + b2)
            return float(np.tanh((Wm @ h + bm)[0]))
        return pol

    def quick(w):
        pol = np_pol(w)
        rows = [run_episode(env, pol, th, al) for env in qenvs for th, al in quick_starts]
        nom = rows[:len(quick_starts)]
        return sum(r["strict"] for r in rows), float(np.mean([r["ret"] for r in nom]))

    class CB(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.train, self.mark, self.next = 0.0, None, every
            self.points, self.stopped, self.best = [], None, None

        def _on_training_start(self):
            self.mark = time.perf_counter()

        def _on_step(self):
            t = self.train + (time.perf_counter() - self.mark)
            if t < self.next and t < budget:
                return True
            self.train = t
            w = weights()
            gate, ret = quick(w)
            self.points.append(dict(t=round(t, 1), steps=int(self.num_timesteps), gate=gate, ret=round(ret, 2)))
            if self.best is None or (gate, ret) >= (self.best[0], self.best[1]):
                self.best = (gate, ret, w, round(t, 1))
            print("%s t=%6.1f s  steps=%7d  시험 %2d/32  점수 %8.1f" % (tag, t, self.num_timesteps, gate, ret), flush=True)
            self.next = t + every
            self.mark = time.perf_counter()
            g = [p["gate"] for p in self.points]
            if len(g) >= 2 and g[-1] >= 30 and g[-2] >= 30:
                self.stopped = "rule"
                return False
            if t >= budget:
                self.stopped = "budget"
                return False
            return True

    cb = CB()
    model.learn(total_timesteps=10 ** 10, callback=cb, progress_bar=False)
    venv.close()
    gate, ret, w, t_best = cb.best
    np.savez(os.path.join(out, tag + ".npz"), **w)
    full = evaluate(np_pol(w), verbose=False)
    rec = dict(key="s%de%d" % (s, e), name=tag, smooth=s, effort=e, w_smooth=ws, w_effort=we,
               label="빠르기 %s, 힘 %s" % (WORD[s], WORD[e]), seed=seed, budget=budget, stopped=cb.stopped,
               t_stop=cb.points[-1]["t"], t_best=t_best, gate32=gate, ret=ret,
               full_strict=full["total_strict"], full_loose=full["total_loose"],
               curve=[p["ret"] for p in cb.points], gates=[p["gate"] for p in cb.points], points=cb.points)
    with open(os.path.join(out, tag + ".json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print("%s 끝: %s %.0f s, 고른 시점 %.0f s, 시험 %d/32, 128칸 %d" % (
        tag, cb.stopped, rec["t_stop"], t_best, gate, full["total_strict"]), flush=True)


def plan(budget, every, out):
    py, me = sys.executable, os.path.abspath(__file__)
    logdir = os.path.join(out, "logs")
    os.makedirs(logdir, exist_ok=True)
    jobs = [(s, e) for s in range(3) for e in range(3)]
    t0 = time.time()
    for i in range(0, len(jobs), 3):
        procs = []
        for s, e in jobs[i:i + 3]:
            cmd = [py, "-u", me, "--s", str(s), "--e", str(e), "--budget", str(budget), "--every", str(every), "--out", out]
            procs.append(subprocess.Popen(cmd, stdout=open(os.path.join(logdir, "s%de%d.log" % (s, e)), "w"),
                                          stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()
        print("묶음 %d 끝 (누적 %.1f 분)" % (i // 3 + 1, (time.time() - t0) / 60), flush=True)
    report(out)


def report(out):
    rows = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(out, "sac_s*e*.json")))]
    print("%-10s %-22s %7s %7s %7s %8s %8s" % ("칸", "규칙", "멈춤", "시각 s", "고른 s", "시험/32", "128칸"))
    for r in rows:
        print("%-10s %-22s %7s %7.0f %7.0f %8d %8d" % (r["name"], r["label"], r["stopped"], r["t_stop"], r["t_best"],
                                                      r["gate32"], r["full_strict"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=1)
    ap.add_argument("--e", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=480.0)
    ap.add_argument("--every", type=float, default=30.0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.report:
        report(a.out)
    elif a.plan:
        plan(a.budget, a.every, a.out)
    else:
        run_one(a.s, a.e, a.seed, a.budget, a.every, a.out)


if __name__ == "__main__":
    main()
