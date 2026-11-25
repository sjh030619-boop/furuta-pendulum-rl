"""바탕 정책 만들기: SAC·PPO 를 같은 환경·같은 멈춤 규칙으로 학습합니다 (재전략 1단계).

    python s2r/base_train.py --cfg ppo_b --seed 0 [--budget 900] [--tag screen]
    python s2r/base_train.py --many ppo_a:0 ppo_b:0 ppo_c:0 ppo_d:0 --tag screen    # 동시에
    python s2r/base_train.py --report [--tag screen]

멈춤 규칙: 학습 30 s 마다 빠른 평가 32칸(조건 8개 x 시작 4개, 엄격 기준). 30/32 이상이 2번 연속이면 멈춤.
멈춘 모델을 128칸으로 평가하고, 120 이상이면 바탕 정책 후보. Pi 호스트 형식 npz 도 같이 저장(export_npz).
SAC 는 runs_stage4/sac_du0_s*.zip 이 이미 같은 규칙으로 학습한 것이라 다시 돌리지 않습니다.

PPO 후보 (모두 64x64 tanh, DummyVecEnv 한 프로세스, torch 1 스레드):
  ppo_a  1단계 설정 그대로: 환경 24, n_steps 256, 배치 1024, 10 epoch, lr 3e-4, 초기 표준편차 1.0, γ 0.99
  ppo_b  조원 B 설정을 SB3 로: 환경 128, n_steps 200, 배치 4096, 6 epoch, lr 1e-4, 초기 표준편차 0.15, γ 0.995
  ppo_c  a + 초기 표준편차 0.37, lr 5e-4
  ppo_d  환경 64, n_steps 128, 배치 2048, 6 epoch, lr 3e-4, 초기 표준편차 0.3, γ 0.995
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
OUT = os.path.join(HERE, "runs_base")

CFG = {
    "ppo_a": dict(n_envs=24, kw=dict(n_steps=256, batch_size=1024, n_epochs=10, learning_rate=3e-4, gamma=0.99),
                  std0=1.0),
    "ppo_b": dict(n_envs=128, kw=dict(n_steps=200, batch_size=4096, n_epochs=6, learning_rate=1e-4, gamma=0.995),
                  std0=0.15),
    "ppo_c": dict(n_envs=24, kw=dict(n_steps=256, batch_size=1024, n_epochs=10, learning_rate=5e-4, gamma=0.99),
                  std0=0.37),
    "ppo_d": dict(n_envs=64, kw=dict(n_steps=128, batch_size=2048, n_epochs=6, learning_rate=3e-4, gamma=0.995),
                  std0=0.3),
    "sac": dict(n_envs=8, kw=None, std0=None),
}


def np_policy(model, algo):
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.policy.state_dict().items()}
    if algo == "sac":
        W1, b1, W2, b2 = (sd["actor.latent_pi.0.weight"], sd["actor.latent_pi.0.bias"],
                          sd["actor.latent_pi.2.weight"], sd["actor.latent_pi.2.bias"])
        Wm, bm = sd["actor.mu.weight"], sd["actor.mu.bias"]

        def pol(o):
            h = np.maximum(0.0, W1 @ o.astype(np.float64) + b1)
            h = np.maximum(0.0, W2 @ h + b2)
            return float(np.tanh((Wm @ h + bm)[0]))
    else:
        W1, b1, W2, b2 = (sd["mlp_extractor.policy_net.0.weight"], sd["mlp_extractor.policy_net.0.bias"],
                          sd["mlp_extractor.policy_net.2.weight"], sd["mlp_extractor.policy_net.2.bias"])
        Wm, bm = sd["action_net.weight"], sd["action_net.bias"]

        def pol(o):
            h = np.tanh(W1 @ o.astype(np.float64) + b1)
            h = np.tanh(W2 @ h + b2)
            return float(np.clip((Wm @ h + bm)[0], -1.0, 1.0))
    return pol


def run_one(cfg, seed, budget, every, out, tag, resume=None, subproc=False):
    import torch
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from s2r_env import make_env, S2REnv, NOMINAL
    from evaluate import evaluate, run_episode, CONDITIONS, STARTS
    from stage1 import SAC_KW, NET
    from stage4_chatter import hold_metrics

    torch.set_num_threads(1)
    c = CFG[cfg]
    algo = "sac" if cfg == "sac" else "ppo"
    name = "%s_s%d" % (cfg, seed)
    fns = [make_env(randomize=True) for _ in range(c["n_envs"])]
    venv = SubprocVecEnv(fns, start_method="spawn") if subproc else DummyVecEnv(fns)
    venv.seed(seed)
    if resume:
        model = (SAC if algo == "sac" else PPO).load(resume, env=venv, device="cpu")
    elif algo == "sac":
        model = SAC("MlpPolicy", venv, gamma=0.99, seed=seed, device="cpu", verbose=0,
                    policy_kwargs=dict(net_arch=NET), **SAC_KW)
    else:
        model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.0, vf_coef=0.5,
                    policy_kwargs=dict(net_arch=dict(pi=NET, vf=NET), log_std_init=math.log(c["std0"])), **c["kw"])
    quick_starts = [STARTS[i] for i in (0, 5, 10, 15)]
    qenvs = []
    for over in CONDITIONS.values():
        e = S2REnv(params=dict(NOMINAL, **over), episode_sec=12.0, init="hang")
        e.reset(seed=0)
        qenvs.append(e)

    def quick():
        pol = np_policy(model, algo)
        return sum(run_episode(e, pol, th, al)["strict"] for e in qenvs for th, al in quick_starts)

    class CB(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.train, self.mark, self.next = 0.0, None, every
            self.points, self.stopped = [], None

        def _on_training_start(self):
            self.mark = time.perf_counter()

        def _on_step(self):
            e = self.train + (time.perf_counter() - self.mark)
            if e < self.next and e < budget:
                return True
            self.train = e
            te = time.perf_counter()
            q = quick()
            self.points.append(dict(t=round(self.train, 1), steps=int(self.num_timesteps), strict=q,
                                    eval_s=round(time.perf_counter() - te, 1)))
            print("%s t=%6.1f s  steps=%8d  빠른 평가 %2d/32" % (name, self.train, self.num_timesteps, q), flush=True)
            self.next = self.train + every
            self.mark = time.perf_counter()
            qs = [p["strict"] for p in self.points]
            if len(qs) >= 2 and qs[-1] >= 30 and qs[-2] >= 30:
                self.stopped = "rule"
                return False
            if self.train >= budget:
                self.stopped = "budget"
                return False
            return True

    cb = CB()
    model.learn(total_timesteps=10 ** 10, callback=cb, progress_bar=False, reset_num_timesteps=not resume)
    venv.close()
    zip_path = os.path.join(out, "%s__%s" % (tag, name))
    model.save(zip_path)
    pol = np_policy(model, algo)
    full = evaluate(pol, verbose=False)
    env = S2REnv(params=dict(NOMINAL), episode_sec=12.0, init="hang")
    env.reset(seed=0)
    iq, dU, rms = hold_metrics(pol, env, [math.radians(a) for a in (30, -30, 15, -15, 5, -5, 2, -2)])
    rec = dict(cfg=cfg, algo=algo, seed=seed, tag=tag, budget=budget, stopped=cb.stopped, resume=resume, subproc=subproc,
               t_stop=cb.points[-1]["t"], steps=int(model.num_timesteps),
               fps=int(model.num_timesteps / max(cb.train, 1e-9)), eval_s=round(sum(p["eval_s"] for p in cb.points), 1),
               full_strict=full["total_strict"], full_loose=full["total_loose"],
               per={k: v["strict"] for k, v in full.items() if isinstance(v, dict)},
               hold_iq=iq, hold_du=dU, hold_rms_deg=rms, points=cb.points, config=dict(n_envs=c["n_envs"], std0=c["std0"],
                                                                                    **(c["kw"] or SAC_KW)))
    rec["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in rec["config"].items()}
    with open(zip_path + ".json", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print("%s 끝: %s %.0f s, 128칸 엄격 %d (느슨 %d), 직립 |iq| %.3f A" % (
        name, cb.stopped, rec["t_stop"], full["total_strict"], full["total_loose"], iq), flush=True)


def report(out, tag):
    rows = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(out, "%s__*.json" % tag)))]
    print("%-8s %4s %7s %7s %9s %9s %8s %9s  %s" % ("설정", "시드", "멈춤", "시각 s", "128 엄격", "128 느슨", "fps", "직립|iq|",
                                                    "빠른 평가"))
    for r in rows:
        print("%-8s %4d %7s %7.0f %9d %9d %8d %7.3f A  %s" % (
            r["cfg"], r["seed"], r["stopped"], r["t_stop"], r["full_strict"], r["full_loose"], r["fps"],
            r["hold_iq"], " ".join(str(p["strict"]) for p in r["points"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", choices=list(CFG))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--every", type=float, default=30.0)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--many", nargs="+")
    ap.add_argument("--resume", help="이어서 학습할 SB3 zip (시각은 이어 붙인 뒤부터 다시 0)")
    ap.add_argument("--subproc", action="store_true", help="환경을 프로세스로 나눠 수집 (PPO 1단계와 같은 방식)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    if a.report:
        report(a.out, a.tag)
    elif a.many:
        procs = []
        for item in a.many:
            cfg, seed = item.split(":")
            cmd = [sys.executable, "-u", os.path.abspath(__file__), "--cfg", cfg, "--seed", seed, "--budget",
                   str(a.budget), "--every", str(a.every), "--tag", a.tag, "--out", a.out]
            log = open(os.path.join(a.out, "logs", "%s__%s_s%s.log" % (a.tag, cfg, seed)), "w")
            procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()
        report(a.out, a.tag)
    else:
        run_one(a.cfg, a.seed, a.budget, a.every, a.out, a.tag, a.resume, a.subproc)


if __name__ == "__main__":
    main()
