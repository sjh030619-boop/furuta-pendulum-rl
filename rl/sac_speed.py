"""SAC를 벽시계 시간 기준으로 조율합니다.

    python rl/sac_speed.py --variant base --seed 0            # 한 번
    python rl/sac_speed.py --sweep screen1 --seeds 0 1 2      # 여러 설정을 순서대로
    python rl/sac_speed.py --report rl/runs_speed             # 표로 요약

출발점은 벤치마크의 SAC 설정 그대로입니다(환경 8개, 배치 256, 64x64, 스텝당 업데이트 1회).
거기서 **한 번에 하나씩** 바꿔 무엇이 실제로 빨라지는지 봅니다.

벤치마크와 다른 점은 평가 간격뿐입니다. 벤치마크는 5초마다 평가해서 "최초 성공"이
5초 눈금으로만 나옵니다(15초 아니면 20초). 조율에는 너무 거칠어 1초마다 평가합니다.
평가 시간은 벤치마크와 같이 학습 시계에서 뺍니다.

풀었다고 보는 시점은 두 가지로 적습니다.
  t_first  성공률 0.8 이상이 처음 나온 시각 (벤치마크의 "최초 성공"과 같은 정의)
  t_sust   그때부터 연속 3번 평가에서 0.8 이상을 유지한 첫 시각. 1초 눈금에서는 우연히
           한 번 닿는 일이 생기므로 판단은 이쪽으로 합니다.

한 번에 한 프로세스만 돌립니다. 여러 설정을 돌릴 때는 시드를 바깥 고리로 두어, 기계 상태가
도중에 변해도 특정 설정에만 몰리지 않게 합니다.
"""
import time

T_START = time.perf_counter()

import os
import sys
import json
import glob
import argparse
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BASE_KW = dict(learning_rate=3e-4, buffer_size=400_000, batch_size=256, tau=0.005,
               train_freq=(1, "step"), gradient_steps=1, learning_starts=2000)

VARIANTS = {
    "base": dict(),
    "b512": dict(kw=dict(batch_size=512)),
    "b1024": dict(kw=dict(batch_size=1024)),
    "b2048": dict(kw=dict(batch_size=2048)),
    "lr1e-3": dict(kw=dict(learning_rate=1e-3)),
    "lr3e-3": dict(kw=dict(learning_rate=3e-3)),
    "n4": dict(n_envs=4),
    "n16": dict(n_envs=16),
    "n32": dict(n_envs=32),
    "g2": dict(kw=dict(gradient_steps=2)),
    "tau02": dict(kw=dict(tau=0.02)),
    "ls500": dict(kw=dict(learning_starts=500)),
    "ent01": dict(kw=dict(ent_coef="auto_0.1")),
    "nstep3": dict(kw=dict(n_steps=3)),
    "ln": dict(ln=True),
    "qf256": dict(qf=[256, 256]),
    "lr6e-4": dict(kw=dict(learning_rate=6e-4)),
    "lr1.5e-3": dict(kw=dict(learning_rate=1.5e-3)),
    "lr2e-3": dict(kw=dict(learning_rate=2e-3)),
    "lr1e-3_n4": dict(n_envs=4, kw=dict(learning_rate=1e-3)),
    "lr1e-3_n2": dict(n_envs=2, kw=dict(learning_rate=1e-3)),
    "lr1e-3_n1": dict(n_envs=1, kw=dict(learning_rate=1e-3)),
    "lr1e-3_b512": dict(kw=dict(learning_rate=1e-3, batch_size=512)),
    "lr2e-3_b512": dict(kw=dict(learning_rate=2e-3, batch_size=512)),
    "lr3e-3_b1024": dict(kw=dict(learning_rate=3e-3, batch_size=1024)),
    "lr1e-3_ln": dict(ln=True, kw=dict(learning_rate=1e-3)),
    "lr3e-3_ln": dict(ln=True, kw=dict(learning_rate=3e-3)),
    "lr1e-3_ls500": dict(kw=dict(learning_rate=1e-3, learning_starts=500)),
}

SWEEPS = {
    "screen1": ["base", "b512", "b1024", "b2048", "lr1e-3", "lr3e-3", "n4", "n16", "n32",
                "g2", "tau02", "ls500", "ent01", "nstep3", "ln", "qf256"],
    "screen2": ["lr6e-4", "lr1.5e-3", "lr2e-3", "lr1e-3_n4", "lr1e-3_n2", "lr1e-3_n1",
                "lr1e-3_b512", "lr2e-3_b512", "lr3e-3_b1024", "lr1e-3_ln", "lr3e-3_ln",
                "lr1e-3_ls500"],
}


def register(name, spec):
    VARIANTS[name] = spec


def make_policy_class(ln):
    """비평가에만 LayerNorm 을 넣은 정책. 배우는 그대로라 앱으로 내보내는 신경망은 같습니다."""
    from torch import nn
    from stable_baselines3.sac.policies import SACPolicy
    from stable_baselines3.common.policies import ContinuousCritic
    from stable_baselines3.common.torch_layers import create_mlp
    from stable_baselines3.common.preprocessing import get_action_dim

    if not ln:
        return "MlpPolicy"

    class LNCritic(ContinuousCritic):
        def __init__(self, observation_space, action_space, net_arch, features_extractor,
                     features_dim, activation_fn=nn.ReLU, normalize_images=True,
                     n_critics=2, share_features_extractor=True):
            super().__init__(observation_space, action_space, net_arch, features_extractor,
                             features_dim, activation_fn, normalize_images, n_critics,
                             share_features_extractor)
            action_dim = get_action_dim(action_space)
            self.q_networks = []
            for idx in range(n_critics):
                q = nn.Sequential(*create_mlp(features_dim + action_dim, 1, net_arch,
                                              activation_fn,
                                              post_linear_modules=[nn.LayerNorm]))
                setattr(self, "qf%d" % idx, q)
                self.q_networks.append(q)

    class LNSACPolicy(SACPolicy):
        def make_critic(self, features_extractor=None):
            kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
            return LNCritic(**kw).to(self.device)

    return LNSACPolicy


def first_times(points, thr=0.8, hold=3):
    t_first = next((p["t"] for p in points if p["success"] >= thr), None)
    t_sust = None
    for i in range(len(points) - hold + 1):
        if all(points[i + k]["success"] >= thr for k in range(hold)):
            t_sust = points[i]["t"]
            break
    return t_first, t_sust


def run_one(variant, seed, budget, every, hold_stop, out, armature=5.0e-3):
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from bench import EVAL_ICS, GAMMA, NET, evaluate, calib_ms
    from furuta_env import FurutaEnv, make_env

    torch.set_num_threads(1)
    spec = VARIANTS[variant]
    n_envs = spec.get("n_envs", 8)
    kw = dict(BASE_KW)
    kw.update(spec.get("kw", {}))
    arch = NET if not spec.get("qf") else dict(pi=NET, qf=spec["qf"])
    kw["policy_kwargs"] = dict(net_arch=arch)
    env_kw = dict(arm_armature=armature)

    venv = DummyVecEnv([make_env(**env_kw) for _ in range(n_envs)])
    venv.seed(seed)
    evs = [FurutaEnv(**env_kw) for _ in EVAL_ICS]
    for j, e in enumerate(evs):
        e.reset(seed=10_000 + 100 * seed + j)
    model = SAC(make_policy_class(spec.get("ln", False)), venv, gamma=GAMMA, seed=seed,
                device="cpu", verbose=0, **kw)

    class CB(BaseCallback):
        """예산 안에서 돌리며 1초마다 평가합니다. 충분히 오래 풀린 채면 일찍 멈춥니다."""

        def __init__(self):
            super().__init__(0)
            self.train_time = 0.0
            self.mark = None
            self.next = 0.0
            self.points = []
            self.run = 0

        def _on_training_start(self):
            self.mark = time.perf_counter()
            self._snap()

        def _snap(self):
            self.train_time += time.perf_counter() - self.mark
            m = evaluate(model, evs, None)
            m["t"] = round(self.train_time, 3)
            m["steps"] = int(self.num_timesteps)
            m["updates"] = int(model._n_updates)
            self.points.append(m)
            self.run = self.run + 1 if m["success"] >= 0.8 else 0
            self.next = self.train_time + every
            self.mark = time.perf_counter()

        def _on_step(self):
            e = self.train_time + (time.perf_counter() - self.mark)
            if e >= budget:
                self._snap()
                return False
            if e >= self.next:
                self._snap()
                if hold_stop and self.run >= hold_stop:
                    return False
            return True

    cb = CB()
    cal0 = calib_ms()
    cpu0 = time.process_time()
    t_learn = time.perf_counter()
    model.learn(total_timesteps=10 ** 9, callback=cb, progress_bar=False)
    own_cpu = time.process_time() - cpu0
    cal1 = calib_ms()
    venv.close()

    t_first, t_sust = first_times(cb.points)
    rec = dict(variant=variant, seed=seed, n_envs=n_envs, budget=budget,
               kw={k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()
                   if k != "policy_kwargs"},
               ln=bool(spec.get("ln", False)), qf=spec.get("qf"),
               train_time=cb.train_time, steps=int(model.num_timesteps),
               updates=int(model._n_updates),
               fps=int(model.num_timesteps / max(cb.train_time, 1e-9)),
               ups=round(model._n_updates / max(cb.train_time, 1e-9), 1),
               t_first=t_first, t_sust=t_sust,
               final_succ=float(np.mean([p["success"] for p in cb.points[-3:]])),
               final_ret=float(np.mean([p["ret"] for p in cb.points[-3:]])),
               setup_time=round(t_learn - T_START, 2),
               calib_ms_before=round(cal0, 1), calib_ms_after=round(cal1, 1),
               own_cpu_share=round(own_cpu / max(cb.train_time, 1e-9), 3),
               stopped_early=bool(cb.train_time < budget - 0.5),
               points=cb.points)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "%s_s%d.json" % (variant, seed)), "w") as f:
        json.dump(rec, f)
    return rec


def fmt(x, pat="%.1f"):
    return "-" if x is None else pat % x


def report(out, names=None):
    rows = {}
    for f in sorted(glob.glob(os.path.join(out, "*.json"))):
        r = json.load(open(f))
        rows.setdefault(r["variant"], []).append(r)
    order = names or sorted(rows, key=lambda k: (
        np.mean([r["t_sust"] if r["t_sust"] is not None else 1e3 for r in rows[k]])))
    base = rows.get("base")
    b = np.mean([r["t_sust"] for r in base if r["t_sust"] is not None]) if base else None
    print("%-14s %3s %6s %8s %8s %14s %7s %7s %7s %6s" % (
        "variant", "n", "ok", "t_first", "t_sust", "시드별 t_sust", "vs base", "fps", "upd/s", "calib"))
    for k in order:
        if k not in rows:
            continue
        v = rows[k]
        ok = [r["t_sust"] for r in v if r["t_sust"] is not None]
        tf = [r["t_first"] for r in v if r["t_first"] is not None]
        m = float(np.mean(ok)) if len(ok) == len(v) else None
        print("%-14s %3d %6s %8s %8s %14s %7s %7.0f %7.0f %6.0f" % (
            k, len(v), "%d/%d" % (len(ok), len(v)),
            fmt(float(np.mean(tf)) if len(tf) == len(v) else None), fmt(m),
            "/".join("%.0f" % t if t is not None else "x" for t in
                     [r["t_sust"] for r in sorted(v, key=lambda r: r["seed"])]),
            ("%+.0f%%" % (100 * (m / b - 1))) if (m is not None and b) else "-",
            np.mean([r["fps"] for r in v]), np.mean([r["ups"] for r in v]),
            max(max(r["calib_ms_before"], r["calib_ms_after"]) for r in v)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=float, default=60.0)
    p.add_argument("--eval-every", type=float, default=1.0)
    p.add_argument("--hold-stop", type=int, default=6)
    p.add_argument("--out", default=os.path.join(HERE, "runs_speed"))
    p.add_argument("--sweep", default="")
    p.add_argument("--names", nargs="*", default=[])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--report", default="")
    p.add_argument("--spec", default="")
    a = p.parse_args()

    if a.spec:
        for k, v in json.loads(a.spec).items():
            register(k, v)

    if a.report:
        report(a.report, a.names or None)
        return

    if a.sweep or a.names:
        names = a.names or SWEEPS[a.sweep]
        n = len(names) * len(a.seeds)
        i = 0
        t0 = time.perf_counter()
        for seed in a.seeds:
            for v in names:
                i += 1
                cmd = [sys.executable, os.path.abspath(__file__), "--variant", v,
                       "--seed", str(seed), "--budget", str(a.budget),
                       "--eval-every", str(a.eval_every), "--hold-stop", str(a.hold_stop),
                       "--out", a.out]
                if a.spec:
                    cmd += ["--spec", a.spec]
                r = subprocess.run(cmd, capture_output=True, text=True)
                line = (r.stdout.strip().splitlines() or ["(출력 없음)"])[-1]
                if r.returncode != 0:
                    line = "FAILED: " + (r.stderr.strip().splitlines() or ["?"])[-1]
                print("[%3d/%3d] %5.1f min  %s" % (i, n, (time.perf_counter() - t0) / 60.0, line),
                      flush=True)
        print()
        report(a.out, names)
        return

    r = run_one(a.variant, a.seed, a.budget, a.eval_every, a.hold_stop, a.out)
    print("%-12s s%d  t_first=%-5s t_sust=%-5s fps=%-5d upd/s=%-5.0f final=%.2f/%.0f  calib %.0f->%.0f"
          % (r["variant"], r["seed"], fmt(r["t_first"]), fmt(r["t_sust"]), r["fps"], r["ups"],
             r["final_succ"], r["final_ret"], r["calib_ms_before"], r["calib_ms_after"]))


if __name__ == "__main__":
    main()
