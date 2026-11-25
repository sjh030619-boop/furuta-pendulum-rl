"""Roll out a saved policy and print what it actually does."""
import os
import sys
import math
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from furuta_env import FurutaEnv
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from sb3_contrib import TQC, CrossQ

CLS = dict(ppo=PPO, a2c=A2C, sac=SAC, td3=TD3, ddpg=DDPG, tqc=TQC, crossq=CrossQ)

p = argparse.ArgumentParser()
p.add_argument("--algo", required=True)
p.add_argument("--path", required=True)
p.add_argument("--alpha0", type=float, default=0.0)
p.add_argument("--armature", type=float, default=5.0e-3)
a = p.parse_args()

env = FurutaEnv(arm_armature=a.armature)
model = CLS[a.algo].load(a.path, device="cpu")
obs = env.set_state(0.0, a.alpha0, 0.0, 0.0)
print("  t[s]  alpha[deg]  phi[deg]  ad[r/s]  theta[deg]  td[r/s]   u      r")
tot = 0.0
for i in range(env.max_steps):
    act, _ = model.predict(obs[None, :], deterministic=True)
    obs, r, term, trunc, info = env.step(act[0])
    tot += r
    al = env.data.qpos[1]
    phi = math.atan2(math.sin(al - math.pi), math.cos(al - math.pi))
    if i % 20 == 0 or term:
        print("%6.3f  %9.1f  %8.1f  %7.2f  %9.1f  %7.2f  %5.2f  %6.3f"
              % (env.data.time, math.degrees(al), math.degrees(phi), env.data.qvel[1],
                 math.degrees(env.data.qpos[0]), env.data.qvel[0], float(act[0][0]), r))
    if term:
        print("TERMINATED at step", i)
        break
print("return = %.1f  max|ad| seen" % tot)
