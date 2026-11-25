"""지연에 견디는 SAC 를 어떻게 학습시킬지 비교합니다.

    python rl/delay_rl.py --train --seeds 0       # 여섯 방식을 시드 0 으로 학습
    python rl/delay_rl.py --eval                  # 저장된 정책을 지연 0~60 ms 에서 평가

벤치마크의 SAC 는 구동 지연 20 ms 에서 성공률이 0% 입니다(`robust_table.txt`). 두 가지를
따로, 그리고 같이 바꿔 봅니다.

  학습 때의 지연     없음 / 20 ms 고정 / 에피소드마다 0~40 ms 무작위
  관측의 행동 이력   없음(6차원) / 직전 명령 2개(8차원, 앱의 관측과 같은 모양)

  plain      지연 없음, 이력 없음   <- 벤치마크와 같은 방식
  d20        20 ms 고정, 이력 없음
  rand       0~40 ms 무작위, 이력 없음
  hist       지연 없음, 이력 있음
  hist_d20   20 ms 고정, 이력 있음
  hist_rand  0~40 ms 무작위, 이력 있음

SAC 설정은 벤치마크 그대로에 학습률만 1e-3 입니다(`sac_speed.py` 에서 확인한 값).

학습 예산은 **시간이 아니라 전이 수**(기본 20만)로 줍니다. 이 실험은 "얼마나 빨리"가 아니라
"되느냐"를 보는 것이고, 전이 수로 고정하면 여러 프로세스를 동시에 돌려도 결과가 달라지지
않습니다. 그래서 여기 찍히는 시간은 벽시계 비교에 쓰면 안 됩니다.

평가는 `robust.py` 와 같은 방식입니다. 초기조건 15개(고정 5개 x 진자 각속도 -2/0/+2),
5초 에피소드의 마지막 2초를 도립으로 유지하면 성공. 센서는 두 가지로 봅니다.
  ideal  각도·각속도를 그대로
  real   12비트 양자화 + 각도 차분으로 구한 속도 (AS5600 과 같은 조건)
"""
import os
import sys
import json
import glob
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODES = {
    "plain": dict(hist=0),
    "d20": dict(hist=0, delay_ms=20.0),
    "rand": dict(hist=0, delay_range_ms=(0.0, 40.0)),
    "hist": dict(hist=2),
    "hist_d20": dict(hist=2, delay_ms=20.0),
    "hist_rand": dict(hist=2, delay_range_ms=(0.0, 40.0)),
}
ORDER = ["plain", "d20", "rand", "hist", "hist_d20", "hist_rand"]

SAC_KW = dict(learning_rate=1e-3, buffer_size=400_000, batch_size=256, tau=0.005,
              train_freq=(1, "step"), gradient_steps=1, learning_starts=2000)

DELAYS = [0, 4, 10, 16, 20, 30, 40, 50, 60]
REAL = dict(bits=12, fd_vel=True)
USB = dict(delay_ms=6.0, jitter_ms=18.0)


def wide_ics():
    from bench import EVAL_ICS
    return [(th, al, ad) for th, al in EVAL_ICS for ad in (-2.0, 0.0, 2.0)]


def rollout(policy, hist, ics, delay_ms=0.0, jitter_ms=0.0, sensor=None, seed=0,
            episode_sec=5.0, tail_sec=2.0):
    """초기조건 전부를 한꺼번에 굴립니다. (마지막 구간 도립 비율, 리턴)을 돌려줍니다."""
    from delay_env import DelayEnv
    from robust import Sensor

    sensor = sensor or {}
    k = len(ics)
    envs = [DelayEnv(episode_sec=episode_sec, delay_ms=delay_ms, jitter_ms=jitter_ms, hist=hist)
            for _ in range(k)]
    for j, e in enumerate(envs):
        e.reset(seed=seed * 1000 + j)
    dt = envs[0].ctrl_dt
    n = envs[0].max_steps
    sens = Sensor(k, dt, sensor.get("bits", 0), sensor.get("noise_deg", 0.0),
                  sensor.get("fd_vel", False), np.random.default_rng(seed))
    obs = np.stack([e.set_state(th0, al0, 0.0, ad0) for e, (th0, al0, ad0) in zip(envs, ics)])
    tail = int(round(tail_sec / dt))
    upr = np.zeros((k, n))
    rets = np.zeros(k)
    alive = np.ones(k, dtype=bool)
    for i in range(n):
        acts, _ = policy.predict(sens(obs), deterministic=True)
        for j, e in enumerate(envs):
            if not alive[j]:
                continue
            o, r, term, _tr, info = e.step(acts[j])
            obs[j] = o
            rets[j] += r
            upr[j, i] = info["upright"]
            if term:
                alive[j] = False
    return upr[:, -tail:].mean(axis=1), rets


def succ(policy, hist, ics, **kw):
    s, r = rollout(policy, hist, ics, **kw)
    return float(np.mean(s > 0.9)), float(np.mean(r))


def train_one(mode, seed, steps, every, out, models):
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from bench import EVAL_ICS, GAMMA, NET
    from delay_env import make_delay_env

    torch.set_num_threads(1)
    spec = MODES[mode]
    venv = DummyVecEnv([make_delay_env(**spec) for _ in range(8)])
    venv.seed(seed)
    model = SAC("MlpPolicy", venv, gamma=GAMMA, seed=seed, device="cpu", verbose=0,
                policy_kwargs=dict(net_arch=NET), **SAC_KW)
    ics = [(th, al, 0.0) for th, al in EVAL_ICS]
    hist = spec["hist"]

    class CB(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.next = 0
            self.points = []

        def _snap(self):
            p = dict(steps=int(self.num_timesteps), updates=int(model._n_updates))
            for ms in (0, 20, 40):
                s, r = succ(model, hist, ics, delay_ms=float(ms))
                p["s%d" % ms] = s
                p["r%d" % ms] = r
            self.points.append(p)
            self.next = self.num_timesteps + every

        def _on_training_start(self):
            self._snap()

        def _on_step(self):
            if self.num_timesteps >= self.next:
                self._snap()
            return self.num_timesteps < steps

    cb = CB()
    t0 = time.perf_counter()
    model.learn(total_timesteps=10 ** 9, callback=cb, progress_bar=False)
    cb._snap()
    wall = time.perf_counter() - t0
    venv.close()

    os.makedirs(models, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    model.save(os.path.join(models, "%s_s%d" % (mode, seed)))
    rec = dict(mode=mode, seed=seed, hist=hist, spec={k: (list(v) if isinstance(v, tuple) else v)
                                                      for k, v in spec.items()},
               steps=int(model.num_timesteps), updates=int(model._n_updates),
               wall_not_comparable=round(wall, 1), points=cb.points)
    with open(os.path.join(out, "%s_s%d.json" % (mode, seed)), "w") as f:
        json.dump(rec, f)
    last = cb.points[-1]
    print("%-10s s%d  %d steps  성공률 0/20/40 ms = %.2f / %.2f / %.2f   (%.0f s)"
          % (mode, seed, rec["steps"], last["s0"], last["s20"], last["s40"], wall), flush=True)


def matrix(policy, hist):
    ics = wide_ics()
    out = dict(ideal={}, real={})
    for ms in DELAYS:
        out["ideal"][str(ms)] = succ(policy, hist, ics, delay_ms=float(ms))[0]
        out["real"][str(ms)] = succ(policy, hist, ics, delay_ms=float(ms), sensor=REAL)[0]
    out["usb"] = float(np.mean([succ(policy, hist, ics, sensor=REAL, seed=s, **USB)[0]
                                for s in (0, 1, 2)]))
    return out


def evaluate_all(models, out, bench_models):
    from stable_baselines3 import SAC
    from furuta_env import FurutaEnv
    from robust import ClassicalPolicy

    res = {}

    def show(name, m):
        print("%-16s ideal: %s" % (name, " ".join("%3.0f" % (100 * m["ideal"][str(d)]) for d in DELAYS)))
        print("%-16s real : %s   USB %3.0f" % ("", " ".join(
            "%3.0f" % (100 * m["real"][str(d)]) for d in DELAYS), 100 * m["usb"]), flush=True)

    print("%-16s        %s" % ("지연 [ms]", " ".join("%3d" % d for d in DELAYS)))
    cp = ClassicalPolicy(FurutaEnv())
    res["LQR+energy"] = dict(kind="classical", hist=0, **matrix(cp, 0))
    show("LQR+energy", res["LQR+energy"])
    for f in sorted(glob.glob(os.path.join(bench_models, "sac_s*.zip"))):
        name = "bench_" + os.path.basename(f)[:-4]
        res[name] = dict(kind="bench", hist=0, **matrix(SAC.load(f, device="cpu"), 0))
        show(name, res[name])
    for mode in ORDER:
        for f in sorted(glob.glob(os.path.join(models, "%s_s*.zip" % mode))):
            name = os.path.basename(f)[:-4]
            hist = MODES[mode]["hist"]
            res[name] = dict(kind="trained", mode=mode, hist=hist,
                             **matrix(SAC.load(f, device="cpu"), hist))
            show(name, res[name])
    with open(out, "w") as f:
        json.dump(dict(delays=DELAYS, usb=USB, results=res), f, indent=1)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--modes", nargs="+", default=ORDER)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--every", type=int, default=10_000)
    p.add_argument("--out", default=os.path.join(HERE, "runs_delay"))
    p.add_argument("--models", default=os.path.join(HERE, "models_delay"))
    p.add_argument("--bench-models", default=os.path.join(HERE, "models"))
    p.add_argument("--eval-out", default=os.path.join(HERE, "delay_eval.json"))
    a = p.parse_args()

    if a.train:
        for seed in a.seeds:
            for mode in a.modes:
                train_one(mode, seed, a.steps, a.every, a.out, a.models)
    if a.eval:
        evaluate_all(a.models, a.eval_out, a.bench_models)


if __name__ == "__main__":
    main()
