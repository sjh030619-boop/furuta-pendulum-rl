"""앱에 넣을 정책을 규칙 조합마다 하나씩 학습합니다.

학습지 3쪽의 두 칸("부드럽게", "힘 아끼기")이 각각 없음/약하게/강하게이므로 조합은 아홉 개입니다.
보상의 가중치만 다르고 나머지는 벤치마크와 완전히 같습니다. furuta_env.py 는 건드리지 않고
서브클래스에서 보상만 다시 계산하므로, 벤치마크 결과의 재현성이 그대로 유지됩니다.

    python rl/train_rules.py --budget 60
"""
import os
import sys
import json
import math
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from furuta_env import FurutaEnv
from bench import EVAL_ICS, GAMMA, NET, evaluate

W_SMOOTH = [0.0, 0.002, 0.008]
W_EFFORT = [0.0, 0.02, 0.10]
WORD = ["없음", "약하게", "강하게"]
TWO_PI = 2.0 * math.pi


HIST = 2


class RuleEnv(FurutaEnv):
    """보상 가중치만 바꾼 환경 + 실기 대비 랜덤화.

    측정으로 확인한 것: 지연을 랜덤화하는 것만으로는 부족하고, 정책이 **자기가 방금 낸 명령**을
    볼 수 있어야 20 ms 지연을 견딥니다(지연이 있으면 진자 각도만으로는 상태가 불완전하기 때문).
    그래서 관측 끝에 최근 명령 두 개를 붙여 6 → 8개가 됩니다.

      지연 0~2스텝(0~40 ms), 각도 잡음 0~0.6°를 에피소드마다 새로 뽑습니다.
    """

    def __init__(self, w_smooth=0.002, w_effort=0.02, dr=True, hist=HIST, **kw):
        self.w_smooth = float(w_smooth)
        self.w_effort = float(w_effort)
        self.dr = dr
        self.hist = hist
        self._uh = [0.0] * max(hist, 1)
        super().__init__(action_delay_steps=2 if dr else 0, **kw)
        if hist:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6 + hist,), dtype=np.float32)

    def _obs(self):
        b = FurutaEnv._obs(self)
        if not self.hist:
            return b
        out = np.empty(6 + self.hist, dtype=np.float32)
        out[:6] = b
        out[6:] = self._uh[-self.hist:]
        return out

    def reset(self, *, seed=None, options=None):
        FurutaEnv.reset(self, seed=seed, options=options)
        if self.dr:
            self.delay = int(self.np_random.integers(0, 3))
            self.obs_noise = math.radians(float(self.np_random.uniform(0.0, 0.6)))
        self._queue = [0.0] * self.delay
        self._uh = [0.0] * max(self.hist, 1)
        return self._obs(), {}

    def set_state(self, theta, alpha, theta_dot=0.0, alpha_dot=0.0):
        self._uh = [0.0] * max(self.hist, 1)
        return FurutaEnv.set_state(self, theta, alpha, theta_dot, alpha_dot)

    def step(self, action):
        u = float(action[0])
        if u < -1.0:
            u = -1.0
        elif u > 1.0:
            u = 1.0
        self._uh.append(u)
        if len(self._uh) > max(self.hist, 1):
            self._uh.pop(0)
        obs, _r, term, trunc, info = FurutaEnv.step(self, action)
        al = self._qpos[1]
        thd = self._qvel[0]
        ald = self._qvel[1]
        phi = (al - math.pi) % TWO_PI
        if phi > math.pi:
            phi -= TWO_PI
        near = math.exp(-11.111111 * phi * phi)
        slow = math.exp(-0.25 * ald * ald)
        r = (1.0 - 0.3 * phi * phi - self.w_smooth * ald * ald - 0.002 * thd * thd
             - self.w_effort * u * u + 0.5 * near * slow)
        return obs, r, term, trunc, info


def make(w_s, w_e, **kw):
    def _f():
        return RuleEnv(w_smooth=w_s, w_effort=w_e, **kw)
    return _f


class Budget(BaseCallback):
    """예산 안에서 돌리면서, 평가가 가장 좋았던 순간의 정책을 따로 저장합니다.
    마지막 정책을 쓰면 중간에 무너진 규칙에서 엉뚱한 정책이 앱에 들어갑니다."""

    def __init__(self, budget, every, box, envs, save_path=None):
        super().__init__(0)
        self.budget = budget
        self.every = every
        self.box = box
        self.envs = envs
        self.save_path = save_path
        self.best = -1e18
        self.train_time = 0.0
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
        if self.save_path and (m["success"] * 1e6 + m["ret"]) > self.best:
            self.best = m["success"] * 1e6 + m["ret"]
            self.box["model"].save(self.save_path)
            self.best_point = dict(m)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=60.0)
    p.add_argument("--eval-every", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--armature", type=float, default=5.0e-3)
    p.add_argument("--models", default=os.path.join(HERE, "models_rules"))
    p.add_argument("--runs", default=os.path.join(HERE, "runs_rules"))
    a = p.parse_args()

    torch.set_num_threads(1)
    os.makedirs(a.models, exist_ok=True)
    os.makedirs(a.runs, exist_ok=True)
    env_kw = dict(arm_armature=a.armature)
    t_all = time.perf_counter()

    for si in range(3):
        for ei in range(3):
            key = "sac_s%de%d" % (si, ei)
            ws, we = W_SMOOTH[si], W_EFFORT[ei]
            venv = DummyVecEnv([make(ws, we, **env_kw) for _ in range(8)])
            venv.seed(a.seed)
            model = SAC("MlpPolicy", venv, gamma=GAMMA, seed=a.seed, device="cpu", verbose=0,
                        learning_rate=3e-4, buffer_size=400_000, batch_size=256, tau=0.005,
                        train_freq=(1, "step"), gradient_steps=1, learning_starts=2000,
                        policy_kwargs=dict(net_arch=NET))
            evs = [RuleEnv(w_smooth=ws, w_effort=we, dr=False, **env_kw) for _ in EVAL_ICS]
            for j, e in enumerate(evs):
                e.reset(seed=50_000 + j)
            path = os.path.join(a.models, key)
            cb = Budget(a.budget, a.eval_every, {"model": model}, evs, save_path=path)
            model.learn(total_timesteps=10 ** 9, callback=cb)
            if cb.best <= -1e17:
                model.save(path)
            last = getattr(cb, "best_point", cb.points[-1])
            solved = next((q["t"] for q in cb.points if q["success"] >= 0.8), None)
            rec = dict(algo="sac", seed=a.seed, tag="", key=key,
                       w_smooth=ws, w_effort=we,
                       label="빠르기 %s, 힘 %s" % (WORD[si], WORD[ei]),
                       budget=a.budget, train_time=cb.train_time, steps=int(model.num_timesteps),
                       fps=int(model.num_timesteps / max(cb.train_time, 1e-9)),
                       device="cpu", n_envs=8, armature=a.armature, task="swingup",
                       ctrl_hz=1.0 / evs[0].ctrl_dt, timestep=float(evs[0].model.opt.timestep),
                       episode_steps=evs[0].max_steps, points=cb.points)
            with open(os.path.join(a.runs, key + ".json"), "w") as f:
                json.dump(rec, f)
            print("%-10s smooth=%-6.3f effort=%-5.2f  solved=%-7s best_succ=%.2f best_ret=%.0f steps=%d"
                  % (key, ws, we, ("%.0fs" % solved) if solved is not None else "never",
                     last["success"], last["ret"], model.num_timesteps))
            venv.close()

    print("total %.1f min" % ((time.perf_counter() - t_all) / 60.0))


if __name__ == "__main__":
    main()
