"""1단계 시뮬 예선: 통일 환경에서 PPO 와 SAC 를 같은 벽시계 예산으로 학습해 비교합니다.

    python s2r/stage1.py --algo sac --seed 0 --budget 1500      # 한 번
    python s2r/stage1.py --plan                                  # 전체 일정(SAC 3개 동시 -> PPO 3개 차례)
    python s2r/stage1.py --report                                # 표

환경: S2REnv(randomize=True, 8 s 에피소드). 판정: evaluate.py (엄격 기준, 128칸).
설정은 이 저장소에서 검증된 값을 그대로 씁니다.
    SAC  환경 8개(Dummy), 학습률 1e-3, 배치 256, 64x64, 스텝당 업데이트 1  (sac_speed.py)
    PPO  환경 24개(Subproc), n_steps 256, 배치 1024, 10 epoch, 학습률 3e-4, 64x64  (bench.py)
학습 중 1분(학습 시간)마다 공칭 조건 16칸을 빠르게 평가합니다. 평가 시간은 학습 시계에서 뺍니다.
예산이 끝난 시점의 모델로 128칸 최종 평가를 합니다(좋은 체크포인트를 골라 쓰지 않음).
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
OUT = os.path.join(HERE, "runs_stage1")

SAC_KW = dict(learning_rate=1e-3, buffer_size=400_000, batch_size=256, tau=0.005,
              train_freq=(1, "step"), gradient_steps=1, learning_starts=2000)
PPO_KW = dict(n_steps=256, batch_size=1024, n_epochs=10, learning_rate=3e-4, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.0, vf_coef=0.5)
NET = [64, 64]
PASS = 12


def run_one(algo, seed, budget, every, out):
    import torch
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from bench import calib_ms
    from s2r_env import make_env
    from evaluate import evaluate

    torch.set_num_threads(1)
    os.makedirs(out, exist_ok=True)
    ckdir = os.path.join(out, "ck_%s_s%d" % (algo, seed))
    os.makedirs(ckdir, exist_ok=True)
    if algo == "sac":
        venv = DummyVecEnv([make_env(randomize=True) for _ in range(8)])
        venv.seed(seed)
        model = SAC("MlpPolicy", venv, gamma=0.99, seed=seed, device="cpu", verbose=0,
                    policy_kwargs=dict(net_arch=NET), **SAC_KW)
    else:
        venv = SubprocVecEnv([make_env(randomize=True) for _ in range(24)], start_method="spawn")
        venv.seed(seed)
        model = PPO("MlpPolicy", venv, gamma=0.99, seed=seed, device="cpu", verbose=0,
                    policy_kwargs=dict(net_arch=dict(pi=NET, vf=NET)), **PPO_KW)
    setup = time.perf_counter() - T0

    def pol(o):
        return float(model.predict(o, deterministic=True)[0][0])

    class CB(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.train = 0.0
            self.mark = None
            self.next = 0.0
            self.points = []
            self.eval_s = 0.0

        def _snap(self):
            self.train += time.perf_counter() - self.mark
            te = time.perf_counter()
            r = evaluate(pol, conditions={"nom": {}}, verbose=False)["nom"]
            self.eval_s += time.perf_counter() - te
            p = dict(t=round(self.train, 1), steps=int(self.num_timesteps), strict=r["strict"],
                     loose=r["loose"], ret=round(r["ret"], 1), term=r["terminated"])
            self.points.append(p)
            model.save(os.path.join(ckdir, "t%05d" % int(self.train)))
            print("%s s%d  t=%6.1f s  steps=%8d  엄격 %2d/16  느슨 %2d/16  리턴 %7.1f" % (
                algo, seed, p["t"], p["steps"], p["strict"], p["loose"], p["ret"]), flush=True)
            self.next = self.train + every
            self.mark = time.perf_counter()

        def _on_training_start(self):
            self.mark = time.perf_counter()
            self._snap()

        def _on_step(self):
            e = self.train + (time.perf_counter() - self.mark)
            if e >= budget:
                self._snap()
                return False
            if e >= self.next:
                self._snap()
            return True

    cal0 = calib_ms()
    cb = CB()
    model.learn(total_timesteps=10 ** 10, callback=cb, progress_bar=False)
    cal1 = calib_ms()
    venv.close()
    model.save(os.path.join(out, "%s_s%d_final" % (algo, seed)))
    print("%s s%d 최종 128칸 평가" % (algo, seed), flush=True)
    full = evaluate(pol, verbose=True)
    pts = cb.points
    t_pass = next((p["t"] for p in pts if p["strict"] >= PASS), None)
    t_sust = None
    for i in range(len(pts) - 2):
        if all(pts[i + j]["strict"] >= PASS for j in range(3)):
            t_sust = pts[i]["t"]
            break
    after = [p["strict"] for p in pts if t_pass is not None and p["t"] >= t_pass]
    rec = dict(algo=algo, seed=seed, budget=budget, setup_s=round(setup, 1), train_s=round(cb.train, 1),
               eval_s=round(cb.eval_s, 1), steps=int(model.num_timesteps),
               fps=int(model.num_timesteps / max(cb.train, 1e-9)),
               calib_ms_before=round(cal0, 1), calib_ms_after=round(cal1, 1),
               t_pass=t_pass, t_sust=t_sust, min_after_pass=min(after) if after else None,
               final_full=full, points=pts,
               config=dict(SAC_KW if algo == "sac" else PPO_KW, n_envs=8 if algo == "sac" else 24))
    rec["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in rec["config"].items()}
    with open(os.path.join(out, "%s_s%d.json" % (algo, seed)), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print("%s s%d 끝: 처음 통과 %s, 유지 통과 %s, 최종 엄격 %d/128, fps %d, calib %.0f->%.0f" % (
        algo, seed, t_pass, t_sust, full["total_strict"], rec["fps"], cal0, cal1), flush=True)


def plan(budget, every, seeds, out):
    py = sys.executable
    me = os.path.abspath(__file__)
    logdir = os.path.join(out, "logs")
    os.makedirs(logdir, exist_ok=True)

    def cmd(algo, s):
        return [py, "-u", me, "--algo", algo, "--seed", str(s), "--budget", str(budget),
                "--every", str(every), "--out", out]

    t0 = time.time()
    print("SAC %d개 동시 시작" % len(seeds), flush=True)
    procs = [subprocess.Popen(cmd("sac", s), stdout=open(os.path.join(logdir, "sac_s%d.log" % s), "w"),
                              stderr=subprocess.STDOUT) for s in seeds]
    for p in procs:
        p.wait()
    print("SAC 끝 (%.1f 분). PPO 차례로 시작" % ((time.time() - t0) / 60), flush=True)
    for s in seeds:
        r = subprocess.run(cmd("ppo", s), stdout=open(os.path.join(logdir, "ppo_s%d.log" % s), "w"),
                           stderr=subprocess.STDOUT)
        print("PPO s%d 끝 (코드 %d, 누적 %.1f 분)" % (s, r.returncode, (time.time() - t0) / 60), flush=True)
    print("전체 끝 %.1f 분" % ((time.time() - t0) / 60), flush=True)
    report(out)


def report(out):
    rows = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(out, "*_s*.json")))]
    print("%-5s %4s %9s %9s %9s %10s %10s %8s %8s %11s" % (
        "알고", "시드", "처음통과", "유지통과", "통과뒤최저", "최종엄격", "최종느슨", "fps", "스텝", "calib"))
    for r in rows:
        f = r["final_full"]
        print("%-5s %4d %9s %9s %9s %7d/128 %7d/128 %8d %8d %5.0f->%-4.0f" % (
            r["algo"], r["seed"], r["t_pass"], r["t_sust"], r["min_after_pass"], f["total_strict"],
            f["total_loose"], r["fps"], r["steps"], r["calib_ms_before"], r["calib_ms_after"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["sac", "ppo"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=float, default=1500.0)
    p.add_argument("--every", type=float, default=60.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default=OUT)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if a.report:
        report(a.out)
    elif a.plan:
        plan(a.budget, a.every, a.seeds, a.out)
    else:
        run_one(a.algo, a.seed, a.budget, a.every, a.out)


if __name__ == "__main__":
    main()
