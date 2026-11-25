"""이미 학습된 정책을 들고 시작하면 초반 학습이 나아지는가 — 측정.

    python rl/warmstart.py --budget 45 --seeds 0 1 2

off-policy 알고리즘에서 "정책을 들고 간다"는 말은 모호합니다. 셋 중 무엇을 들고 가느냐에 따라
결과가 완전히 달라지기 때문에 따로 재야 합니다.

  actor      배우(정책)만. 비평가(Q함수)는 난수에서 시작합니다.
  full       배우 + 비평가 + 그 목표망까지. 재생버퍼는 비어 있습니다.
  full_buf   위에 더해, 불러온 정책으로 먼저 굴려 재생버퍼를 채운 뒤 시작합니다.

그리고 교육용 키트에서 실제로 일어나는 일은 **보상 규칙이 바뀌는 것**입니다. 학생이 규칙을
바꾸면 들고 온 비평가는 **다른 보상**에 대해 학습된 값이라 틀린 값을 가르칩니다. 그 경우도
따로 잽니다(`*_r1`).

버퍼를 채우는 시간도 학습 시계에 포함합니다. 공짜가 아니기 때문입니다.
"""
import os
import sys
import json
import time
import math
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from furuta_env import FurutaEnv
from bench import EVAL_ICS, GAMMA, NET, evaluate, calib_ms

TWO_PI = 2.0 * math.pi

R0 = dict(w_smooth=0.002, w_effort=0.02)
R1 = dict(w_smooth=0.002, w_effort=0.10)
R2 = dict(w_smooth=0.008, w_effort=0.0)
RW = {"_r1": R1, "_r2": R2}


class RwEnv(FurutaEnv):
    """보상 가중치만 바꾼 환경. 관측·행동·물리는 벤치마크와 완전히 같습니다."""

    def __init__(self, w_smooth=0.002, w_effort=0.02, **kw):
        self.w_smooth = float(w_smooth)
        self.w_effort = float(w_effort)
        super().__init__(**kw)

    def step(self, action):
        obs, _r, term, trunc, info = FurutaEnv.step(self, action)
        u = float(np.clip(action[0], -1.0, 1.0))
        al, thd, ald = self._qpos[1], self._qvel[0], self._qvel[1]
        phi = (al - math.pi) % TWO_PI
        if phi > math.pi:
            phi -= TWO_PI
        near = math.exp(-11.111111 * phi * phi)
        slow = math.exp(-0.25 * ald * ald)
        r = (1.0 - 0.3 * phi * phi - self.w_smooth * ald * ald - 0.002 * thd * thd
             - self.w_effort * u * u + 0.5 * near * slow)
        return obs, r, term, trunc, info


def make(rw, **kw):
    def _f():
        return RwEnv(**rw, **kw)
    return _f


class Budget(BaseCallback):
    def __init__(self, budget, every, box, envs, t_offset=0.0):
        super().__init__(0)
        self.budget = budget
        self.every = every
        self.box = box
        self.envs = envs
        self.train_time = float(t_offset)
        self.mark = None
        self.next = 0.0
        self.points = []

    def _on_training_start(self):
        self.mark = time.perf_counter()
        self._snap()

    def _snap(self):
        self.train_time += time.perf_counter() - self.mark
        m = evaluate(self.box["model"], self.envs, None)
        m["t"] = round(self.train_time, 3)
        m["steps"] = int(self.num_timesteps)
        self.points.append(m)
        self.next = self.train_time + self.every
        self.mark = time.perf_counter()

    def _on_step(self):
        e = self.train_time + (time.perf_counter() - self.mark)
        if e >= self.budget:
            self._snap()
            return False
        if e >= self.next:
            self._snap()
        return True


SAC_KW = dict(learning_rate=3e-4, buffer_size=400_000, batch_size=256, tau=0.005,
              train_freq=(1, "step"), gradient_steps=1,
              policy_kwargs=dict(net_arch=NET))

PPO_KW = dict(n_steps=256, batch_size=1024, n_epochs=10, learning_rate=3e-4,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
              policy_kwargs=dict(net_arch=dict(pi=NET, vf=NET)))

ALGO = {
    "sac": dict(cls=SAC, n_envs=8, vec="dummy", kw=SAC_KW),
    "ppo": dict(cls=PPO, n_envs=24, vec="subproc", kw=PPO_KW),
}


def copy_actor(dst, src, algo):
    """정책(배우) 쪽 가중치만 옮깁니다. 값함수/Q함수는 난수 그대로 둡니다.

    PPO는 공유 몸통이 없는 구조(net_arch=dict(pi=..., vf=...))라 정책 가지만 떼어 옮길 수
    있습니다. 행동 잡음(log_std)도 정책의 일부이므로 같이 옮깁니다."""
    if algo == "sac":
        dst.policy.actor.load_state_dict(src.policy.actor.state_dict())
        return
    dst.policy.mlp_extractor.policy_net.load_state_dict(
        src.policy.mlp_extractor.policy_net.state_dict())
    dst.policy.action_net.load_state_dict(src.policy.action_net.state_dict())
    with torch.no_grad():
        dst.policy.log_std.copy_(src.policy.log_std)


def prefill(model, venv, n_steps):
    """불러온 정책으로 굴려 재생버퍼를 채웁니다. 걸린 시간을 돌려줍니다."""
    t0 = time.perf_counter()
    obs = venv.reset()
    n = venv.num_envs
    done_total = 0
    while done_total < n_steps:
        act, _ = model.predict(obs, deterministic=False)
        new_obs, rew, dones, infos = venv.step(act)
        real_next = new_obs.copy()
        for i, d in enumerate(dones):
            if d and infos[i].get("terminal_observation") is not None:
                real_next[i] = infos[i]["terminal_observation"]
        model.replay_buffer.add(obs, real_next, act, rew, dones, infos)
        obs = new_obs
        done_total += n
    return time.perf_counter() - t0


def run_one(algo, mode, seed, budget, every, src_dir, env_kw, armature):
    suf = next((x for x in RW if mode.endswith(x)), None)
    rw = RW[suf] if suf else R0
    base = mode[:-len(suf)] if suf else mode

    cfg = ALGO[algo]
    fns = [make(rw, **env_kw) for _ in range(cfg["n_envs"])]
    venv = (SubprocVecEnv(fns, start_method="spawn") if cfg["vec"] == "subproc"
            else DummyVecEnv(fns))
    venv.seed(seed)
    evs = [RwEnv(**rw, **env_kw) for _ in EVAL_ICS]
    for j, e in enumerate(evs):
        e.reset(seed=50_000 + j)

    src_path = os.path.join(src_dir, "%s_s%d.zip" % (algo, seed))
    t_pre = 0.0

    extra = dict(learning_starts=2000) if algo == "sac" else {}

    if base == "scratch":
        model = cfg["cls"]("MlpPolicy", venv, gamma=GAMMA, seed=seed, device="cpu",
                           verbose=0, **extra, **cfg["kw"])
    elif base == "actor":
        model = cfg["cls"]("MlpPolicy", venv, gamma=GAMMA, seed=seed, device="cpu",
                           verbose=0, **extra, **cfg["kw"])
        src = cfg["cls"].load(src_path, device="cpu")
        copy_actor(model, src, algo)
        del src
    elif base in ("full", "full_buf"):
        if base == "full_buf" and algo != "sac":
            raise SystemExit("full_buf is off-policy only")
        model = cfg["cls"].load(src_path, env=venv, device="cpu")
        model.set_random_seed(seed)
        if algo == "sac":
            model.learning_starts = 0
        model.learning_rate = cfg["kw"]["learning_rate"]
        model._setup_lr_schedule()
        if base == "full_buf":
            t_pre = prefill(model, venv, 4000)
    else:
        raise SystemExit("unknown mode " + mode)

    cb = Budget(budget, every, {"model": model}, evs, t_offset=t_pre)
    model.learn(total_timesteps=10 ** 9, callback=cb, reset_num_timesteps=True)
    venv.close()

    pts = cb.points
    solved = next((p["t"] for p in pts if p["success"] >= 0.8), None)
    return dict(algo=algo, mode=mode, seed=seed, armature=armature, budget=budget,
                prefill_s=round(t_pre, 3), train_time=cb.train_time,
                steps=int(model.num_timesteps), t_solve=solved,
                succ0=pts[0]["success"], ret0=pts[0]["ret"],
                final_succ=pts[-1]["success"], final_ret=pts[-1]["ret"],
                points=pts)


MODES = ["scratch", "actor", "full", "full_buf", "scratch_r1", "full_r1",
         "scratch_r2", "full_r2"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    p.add_argument("--budget", type=float, default=45.0)
    p.add_argument("--eval-every", type=float, default=2.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--modes", nargs="+", default=MODES)
    p.add_argument("--armature", type=float, default=5.0e-3)
    p.add_argument("--src", default=os.path.join(HERE, "models"))
    p.add_argument("--out", default=os.path.join(HERE, "runs_warm"))
    a = p.parse_args()

    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    env_kw = dict(arm_armature=a.armature)
    t_all = time.perf_counter()
    print("calib before: %.0f ms" % calib_ms())

    n = len(a.modes) * len(a.seeds)
    i = 0
    for mode in a.modes:
        for seed in a.seeds:
            i += 1
            c0 = calib_ms()
            print("[%2d/%2d] %-11s seed=%d ..." % (i, n, mode, seed), end=" ", flush=True)
            rec = run_one(a.algo, mode, seed, a.budget, a.eval_every, a.src, env_kw, a.armature)
            rec["calib_ms_before"] = c0
            rec["calib_ms_after"] = calib_ms()
            with open(os.path.join(a.out, "%s_s%d.json" % (mode, seed)), "w") as f:
                json.dump(rec, f)
            print("t0_succ=%.2f t0_ret=%4.0f solve=%-7s final=%.2f/%3.0f steps=%d prefill=%.1fs"
                  % (rec["succ0"], rec["ret0"],
                     ("%.0fs" % rec["t_solve"]) if rec["t_solve"] is not None else "never",
                     rec["final_succ"], rec["final_ret"], rec["steps"], rec["prefill_s"]))

    print("calib after: %.0f ms" % calib_ms())
    print("total %.1f min" % ((time.perf_counter() - t_all) / 60.0))


if __name__ == "__main__":
    main()
