"""Wall-clock budgeted RL benchmark on the Furuta swing-up task.

One process trains one (algorithm, seed) pair for a fixed budget of *training* wall-clock
seconds. Evaluation time is excluded from that clock, so the x axis of the resulting plot is
honest training time. Results go to a JSON file per run.

    python rl/bench.py --algo ppo --seed 0 --budget 180
"""
import time

T_START = time.perf_counter()
T_LEARN = T_START

import os
import sys
import json
import math
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import TQC, CrossQ

from furuta_env import FurutaEnv, make_env

GAMMA = 0.99
NET = [64, 64]

EVAL_ICS = [
    (0.0, 0.00),
    (0.0, 0.05),
    (0.0, -0.05),
    (0.15, 0.02),
    (-0.15, -0.02),
]

ONPOLICY = {"ppo", "a2c"}

CONFIGS = {
    "ppo": dict(
        cls=PPO, n_envs=24, vec="subproc",
        kw=dict(n_steps=256, batch_size=1024, n_epochs=10, learning_rate=3e-4,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
                policy_kwargs=dict(net_arch=dict(pi=NET, vf=NET))),
    ),
    "a2c": dict(
        cls=A2C, n_envs=24, vec="subproc",
        kw=dict(n_steps=16, learning_rate=7e-4, gae_lambda=0.95, ent_coef=0.0,
                normalize_advantage=True,
                policy_kwargs=dict(net_arch=dict(pi=NET, vf=NET))),
    ),
    "sac": dict(
        cls=SAC, n_envs=8,
        kw=dict(learning_rate=3e-4, buffer_size=400_000, batch_size=256, tau=0.005,
                train_freq=(1, "step"), gradient_steps=1, learning_starts=2000,
                policy_kwargs=dict(net_arch=NET)),
    ),
    "td3": dict(
        cls=TD3, n_envs=8,
        kw=dict(learning_rate=3e-4, buffer_size=400_000, batch_size=256, tau=0.005,
                train_freq=(1, "step"), gradient_steps=1, learning_starts=2000,
                policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5,
                policy_kwargs=dict(net_arch=NET)),
    ),
    "ddpg": dict(
        cls=DDPG, n_envs=8,
        kw=dict(learning_rate=3e-4, buffer_size=400_000, batch_size=256, tau=0.005,
                train_freq=(1, "step"), gradient_steps=1, learning_starts=2000,
                policy_kwargs=dict(net_arch=NET)),
    ),
    "tqc": dict(
        cls=TQC, n_envs=8,
        kw=dict(learning_rate=3e-4, buffer_size=400_000, batch_size=256, tau=0.005,
                train_freq=(1, "step"), gradient_steps=1, learning_starts=2000,
                top_quantiles_to_drop_per_net=2,
                policy_kwargs=dict(net_arch=NET, n_quantiles=25)),
    ),
    "crossq": dict(
        cls=CrossQ, n_envs=8,
        kw=dict(learning_rate=1e-3, buffer_size=400_000, batch_size=256,
                train_freq=(1, "step"), gradient_steps=1, learning_starts=2000,
                policy_kwargs=dict(net_arch=dict(pi=NET, qf=[256, 256]))),
    ),
}


def evaluate(model, envs, env_kw):
    """Deterministic rollout from a fixed set of initial conditions, all in lockstep so the
    policy is queried once per step for the whole batch."""
    k = len(envs)
    n = envs[0].max_steps
    tail = int(round(2.0 / envs[0].ctrl_dt))
    obs = np.stack([e.set_state(th0, al0, 0.0, 0.0) for e, (th0, al0) in zip(envs, EVAL_ICS)])
    rets = np.zeros(k)
    upr = np.zeros((k, n))
    alive = np.ones(k, dtype=bool)
    for i in range(n):
        acts, _ = model.predict(obs, deterministic=True)
        for j, e in enumerate(envs):
            if not alive[j]:
                continue
            o, r, term, trunc, info = e.step(acts[j])
            obs[j] = o
            rets[j] += r
            upr[j, i] = info["upright"]
            if term:
                alive[j] = False
        if not alive.any():
            break
    ups = upr.mean(axis=1)
    final_ups = upr[:, -tail:].mean(axis=1)
    return dict(
        ret=float(np.mean(rets)),
        ret_min=float(np.min(rets)),
        upright=float(np.mean(ups)),
        settled=float(np.mean(final_ups)),
        success=float(np.mean(final_ups > 0.9)),
    )


class BudgetCallback(BaseCallback):
    def __init__(self, budget, eval_every, model_ref_box, eval_env, env_kw, verbose=0):
        super().__init__(verbose)
        self.budget = budget
        self.eval_every = eval_every
        self.box = model_ref_box
        self.eval_env = eval_env
        self.env_kw = env_kw
        self.train_time = 0.0
        self.mark = None
        self.next_eval = 0.0
        self.points = []

    def _on_training_start(self):
        self.mark = time.perf_counter()
        self._snap()

    def _elapsed(self):
        return self.train_time + (time.perf_counter() - self.mark)

    def _snap(self):
        self.train_time += time.perf_counter() - self.mark
        m = evaluate(self.box["model"], self.eval_env, self.env_kw)
        m["t"] = round(self.train_time, 3)
        m["steps"] = int(self.num_timesteps)
        self.points.append(m)
        self.next_eval = self.train_time + self.eval_every
        self.mark = time.perf_counter()

    def _on_step(self):
        e = self._elapsed()
        if e >= self.budget:
            self._snap()
            return False
        if e >= self.next_eval:
            self._snap()
        return True


def calib_ms(reps=140):
    """Fixed chunk of work, timed. Grows when something else on the machine is eating CPU,
    so it makes a contaminated wall-clock measurement visible after the fact."""
    a = np.random.default_rng(0).standard_normal((256, 256))
    t = time.perf_counter()
    for _ in range(reps):
        a = a @ a / 64.0
    return (time.perf_counter() - t) * 1000.0


def build(algo, seed, device, env_kw, n_envs=None, kw_patch=None, vec="dummy", workers=6):
    cfg = CONFIGS[algo]
    n_envs = n_envs or cfg["n_envs"]
    vec = cfg.get("vec", vec) if vec == "auto" else vec
    kw = dict(cfg["kw"])
    if kw_patch:
        for k, v in kw_patch.items():
            if k == "train_freq" and isinstance(v, list):
                v = tuple(v)
            if k == "net_arch":
                continue
            kw[k] = v
    cfg = dict(cfg, kw=kw)
    fns = [make_env(**env_kw) for _ in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn") if vec == "subproc" else DummyVecEnv(fns)
    venv.seed(seed)
    model = cfg["cls"]("MlpPolicy", venv, gamma=GAMMA, seed=seed, device=device,
                       verbose=0, **cfg["kw"])
    return model, venv, n_envs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=list(CONFIGS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=float, default=180.0)
    p.add_argument("--eval-every", type=float, default=7.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--armature", type=float, default=5.0e-3)
    p.add_argument("--task", default="swingup")
    p.add_argument("--out", default=os.path.join(HERE, "runs"))
    p.add_argument("--tag", default="")
    p.add_argument("--n-envs", type=int, default=0)
    p.add_argument("--kw", default="")
    p.add_argument("--save", default="")
    p.add_argument("--env-kw", default="")
    p.add_argument("--vec", default="auto", choices=["auto", "dummy", "subproc"])
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    env_kw = dict(arm_armature=a.armature, task=a.task)
    if a.env_kw:
        env_kw.update(json.loads(a.env_kw))

    dev = a.device
    if dev == "auto":
        dev = "cpu"

    patch = json.loads(a.kw) if a.kw else None
    model, venv, n_envs = build(a.algo, a.seed, dev, env_kw, a.n_envs or None, patch, a.vec)
    eval_envs = [FurutaEnv(**env_kw) for _ in EVAL_ICS]
    for j, e in enumerate(eval_envs):
        e.reset(seed=10_000 + 100 * a.seed + j)
    eval_env = eval_envs[0]
    box = {"model": model}
    cb = BudgetCallback(a.budget, a.eval_every, box, eval_envs, env_kw)

    cal0 = calib_ms()
    cpu0 = time.process_time()
    global T_LEARN
    T_LEARN = time.perf_counter()
    t0 = T_LEARN
    model.learn(total_timesteps=10**9, callback=cb, progress_bar=False)
    total_wall = time.perf_counter() - t0
    own_cpu = time.process_time() - cpu0
    cal1 = calib_ms()

    os.makedirs(a.out, exist_ok=True)
    rec = dict(
        algo=a.algo, seed=a.seed, device=str(model.device), n_envs=n_envs,
        budget=a.budget, train_time=cb.train_time, total_wall=total_wall,
        steps=int(model.num_timesteps), gamma=GAMMA, net=NET,
        armature=a.armature, task=a.task, ctrl_hz=1.0 / eval_env.ctrl_dt,
        timestep=float(eval_env.model.opt.timestep), decim=eval_env.decim,
        episode_steps=eval_env.max_steps, tag=a.tag,
        setup_time=round(T_LEARN - T_START, 2),
        calib_ms_before=round(cal0, 1), calib_ms_after=round(cal1, 1),
        own_cpu_share=round(own_cpu / max(cb.train_time, 1e-9), 3),
        fps=int(model.num_timesteps / max(cb.train_time, 1e-9)),
        threads=a.threads, kw_patch=patch, vec=a.vec,
        points=cb.points,
    )
    if a.save:
        model.save(a.save)
    name = "%s_s%d%s.json" % (a.algo, a.seed, ("_" + a.tag) if a.tag else "")
    with open(os.path.join(a.out, name), "w") as f:
        json.dump(rec, f, indent=1)
    best = max(pt["ret"] for pt in cb.points)
    print("%-7s seed=%d dev=%s steps=%d fps=%d train=%.1fs best_ret=%.0f final_succ=%.2f"
          % (a.algo, a.seed, model.device, rec["steps"], rec["fps"], cb.train_time,
             best, cb.points[-1]["success"]))
    venv.close()


if __name__ == "__main__":
    main()
