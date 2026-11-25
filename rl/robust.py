"""Robustness of a trained policy: recovery from any state, impulse rejection, and sensor realism.

Nothing here touches furuta_env.py -- the sensor model (AS5600 quantisation, finite-difference
velocity, Gaussian noise) is applied as a wrapper on the observation vector, and the actuation
delay uses the env's own `action_delay_steps` hook. The classical LQR + energy controller is
driven from the SAME observation vector as the agents, so the sensor tests are apples to apples.

    python rl/robust.py --models rl/models --out rl/robust.json
"""
import os
import sys
import json
import math
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from furuta_env import FurutaEnv, W_ARM, W_PEND
from bench import EVAL_ICS
import classical as CL

from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from sb3_contrib import TQC, CrossQ

CLS = dict(ppo=PPO, a2c=A2C, sac=SAC, td3=TD3, ddpg=DDPG, tqc=TQC, crossq=CrossQ)

AS5600_LSB = 2.0 * math.pi / 4096.0

SCENARIOS = [
    ("ideal", dict()),
    ("12bit", dict(bits=12)),
    ("12bit + 20ms delay", dict(bits=12, delay=1)),
    ("12bit + diff vel", dict(bits=12, fd_vel=True)),
    ("12bit + diff vel + 20ms", dict(bits=12, fd_vel=True, delay=1)),
    ("0.5deg noise + 20ms", dict(noise_deg=0.5, delay=1)),
    ("1.0deg noise + 40ms", dict(noise_deg=1.0, delay=2)),
]


class Sensor:
    """Corrupts the 6-vector observation the way the real encoders would."""

    def __init__(self, n, dt, bits=0, noise_deg=0.0, fd_vel=False, rng=None):
        self.q = (2.0 * math.pi / (1 << bits)) if bits else 0.0
        self.sd = math.radians(noise_deg)
        self.fd = fd_vel
        self.dt = dt
        self.rng = rng or np.random.default_rng(0)
        self.prev = np.full((n, 2), np.nan)

    def reset(self):
        self.prev[:] = np.nan

    def __call__(self, obs):
        th = np.arctan2(obs[:, 1], obs[:, 0])
        al = np.arctan2(obs[:, 3], obs[:, 2])
        ang = np.stack([th, al], axis=1)
        if self.sd > 0.0:
            ang = ang + self.rng.normal(0.0, self.sd, size=ang.shape)
        if self.q > 0.0:
            ang = np.round(ang / self.q) * self.q
        out = obs.copy()
        out[:, 0] = np.cos(ang[:, 0])
        out[:, 1] = np.sin(ang[:, 0])
        out[:, 2] = np.cos(ang[:, 1])
        out[:, 3] = np.sin(ang[:, 1])
        if self.fd:
            d = ang - self.prev
            d = (d + math.pi) % (2.0 * math.pi) - math.pi
            v = np.where(np.isnan(self.prev), 0.0, d / self.dt)
            out[:, 4] = v[:, 0] / W_ARM
            out[:, 5] = v[:, 1] / W_PEND
        self.prev = ang
        return out


class ClassicalPolicy:
    """Energy swing-up + LQR, reading the (possibly corrupted) observation vector."""
    name = "LQR+energy"

    def __init__(self, env):
        self.Jp, self.mgl = CL.plant_model(env)
        self.K, _, _, self.rho = CL.lqr_gain(env)
        env.reset(seed=0)
        best = None
        for sign in (1.0, -1.0):
            for kE in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0):
                for phi_sw in (0.25, 0.4, 0.6):
                    for ald_sw in (2.0, 4.0, 8.0):
                        s = CL.score(env, self.K, self.Jp, self.mgl, kE, sign, phi_sw, ald_sw)
                        if best is None or s[0] > best[0][0]:
                            best = (s, (kE, sign, phi_sw, ald_sw))
        self.kE, self.sign, self.phi_sw, self.ald_sw = best[1]
        self.tuned = best[0]

    def predict(self, obs, deterministic=True):
        th = np.arctan2(obs[:, 1], obs[:, 0])
        al = np.arctan2(obs[:, 3], obs[:, 2])
        thd = obs[:, 4] * W_ARM
        ald = obs[:, 5] * W_PEND
        phi = (al - math.pi + math.pi) % (2.0 * math.pi) - math.pi
        lqr = -(self.K[0] * th + self.K[1] * phi + self.K[2] * thd + self.K[3] * ald)
        E = 0.5 * self.Jp * ald * ald + self.mgl * (1.0 - np.cos(al))
        pump = self.sign * self.kE * (2.0 * self.mgl - E) * np.sign(ald * np.cos(al))
        use = (np.abs(phi) < self.phi_sw) & (np.abs(ald) < self.ald_sw)
        u = np.where(use, lqr, pump)
        return np.clip(u, -1.0, 1.0)[:, None].astype(np.float32), None


def batch_run(policy, ics, sensor_kw, episode_sec=5.0, tail_sec=2.0, impulse=None, seed=0):
    """Run every initial condition in lockstep. Returns the settled fraction per rollout."""
    delay = sensor_kw.get("delay", 0)
    k = len(ics)
    envs = [FurutaEnv(episode_sec=episode_sec, action_delay_steps=delay) for _ in range(k)]
    for j, e in enumerate(envs):
        e.reset(seed=seed * 1000 + j)
    dt = envs[0].ctrl_dt
    n = envs[0].max_steps
    sens = Sensor(k, dt, sensor_kw.get("bits", 0), sensor_kw.get("noise_deg", 0.0),
                  sensor_kw.get("fd_vel", False), np.random.default_rng(seed))
    obs = np.stack([e.set_state(th0, al0, 0.0, ad0) for e, (th0, al0, ad0) in zip(envs, ics)])
    tail = int(round(tail_sec / dt))
    kick_at = int(round(impulse[0] / dt)) if impulse else -1
    upr = np.zeros((k, n))
    for i in range(n):
        if i == kick_at:
            for e in envs:
                e.data.qvel[1] += impulse[1]
        acts, _ = policy.predict(sens(obs), deterministic=True)
        for j, e in enumerate(envs):
            o, r, term, trunc, info = e.step(acts[j])
            obs[j] = o
            upr[j, i] = info["upright"]
    return upr[:, -tail:].mean(axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=os.path.join(HERE, "models"))
    p.add_argument("--out", default=os.path.join(HERE, "robust.json"))
    p.add_argument("--classical", action="store_true", default=True)
    p.add_argument("--pick", nargs="*", default=None,
                   help="only test models whose filename stem is in this list")
    a = p.parse_args()

    pols = []
    if a.classical:
        env0 = FurutaEnv()
        cp = ClassicalPolicy(env0)
        print("LQR K = [%.3f %.3f %.3f %.3f], swing-up kE=%.0f sign=%+.0f phi_sw=%.2f ald_sw=%.0f"
              % (cp.K[0], cp.K[1], cp.K[2], cp.K[3], cp.kE, cp.sign, cp.phi_sw, cp.ald_sw))
        pols.append(("LQR+energy", cp))
    for f in sorted(glob.glob(os.path.join(a.models, "*.zip"))):
        base = os.path.basename(f)[:-4]
        algo = base.split("_")[0]
        if algo not in CLS:
            continue
        if a.pick and base not in a.pick:
            continue
        pols.append((base, CLS[algo].load(f, device="cpu")))
    if not pols:
        print("no policies found in", a.models)
        return
    print("policies:", ", ".join(n for n, _ in pols))

    grid_ics = [(0.0, al, ad)
                for al in np.linspace(-math.pi, math.pi, 25)
                for ad in (-6.0, -3.0, 0.0, 3.0, 6.0)]
    std_ics = [(th, al, 0.0) for th, al in EVAL_ICS]
    wide_ics = [(th, al, ad) for th, al in EVAL_ICS for ad in (-2.0, 0.0, 2.0)]
    kicks = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

    out = {}
    for name, pol in pols:
        rec = {}
        g = batch_run(pol, grid_ics, {})
        rec["recovery_grid"] = float(np.mean(g > 0.9))
        rec["recovery_grid_mean_hold"] = float(np.mean(g))
        imp = {}
        for dv in kicks:
            s = batch_run(pol, std_ics, {}, episode_sec=8.0, impulse=(4.0, dv))
            imp["%.0f" % dv] = float(np.mean(s > 0.9))
        rec["impulse"] = imp
        sc = {}
        for label, kw in SCENARIOS:
            vals = [batch_run(pol, wide_ics, kw, seed=s) for s in (0, 1, 2)]
            sc[label] = float(np.mean([np.mean(v > 0.9) for v in vals]))
        rec["sensor"] = sc
        out[name] = rec
        print("\n== %s" % name)
        print("  recovery from state grid (125 ICs): %.0f%%  (mean hold %.2f)"
              % (rec["recovery_grid"] * 100, rec["recovery_grid_mean_hold"]))
        print("  impulse rejection [rad/s]: " + "  ".join(
            "%s:%.0f%%" % (kk, vv * 100) for kk, vv in imp.items()))
        for label, _ in SCENARIOS:
            print("  %-26s %.0f%%" % (label, sc[label] * 100))

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
