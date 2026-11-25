"""Classical reference controller: energy-based swing-up + LQR balance.

The LQR gain is computed from the model, not tuned: MuJoCo's finite-difference transition
Jacobian at the upright equilibrium is converted to the 50 Hz zero-order-hold discrete model,
and the discrete Riccati equation is solved with the SAME quadratic weights the RL reward
uses, so this controller optimises (locally) exactly what the agents are maximising.

The swing-up law is the standard energy-pumping controller; its gain and the hand-off
threshold are picked by a small sweep, which is engineering time, not training time.

    python rl/classical.py --out rl/baseline.json
"""
import os
import sys
import json
import math
import time
import argparse
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from furuta_env import FurutaEnv
from bench import EVAL_ICS
import export_for_app as EXP



def wrap(x):
    y = (x + math.pi) % (2.0 * math.pi) - math.pi
    return y


def plant_model(env):
    """Pendulum inertia about its hinge and the gravity torque scale m*g*l."""
    m, d = env.model, env.data
    mujoco.mj_resetData(m, d)
    d.qpos[:] = [0.0, math.pi * 0.5]
    mujoco.mj_forward(m, d)
    mgl = abs(float(d.qfrc_bias[1]))
    full = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, full)
    return float(full[1, 1]), mgl


def expm(A, terms=40):
    R = np.eye(A.shape[0])
    T = np.eye(A.shape[0])
    for k in range(1, terms):
        T = T @ A / k
        R = R + T
    return R


def lqr_gain(env, q_th=1/1.5**2, q_phi=1/0.35**2, q_thd=1/8.0**2, q_ald=1/6.0**2, r_u=1.0, iters=20000, tol=1e-13):
    """Discrete LQR for x = [theta, phi, theta_dot, alpha_dot] at the control rate.

    The arm angle has to be in the state. The dynamics do not depend on it, so leaving it out
    looks harmless -- but then nothing stops the arm drifting, it winds up to the motor's speed
    limit, and at that speed the motor has no torque left to catch the pendulum with.

    MuJoCo's own transition Jacobian refuses the RK4 integrator, so the linearisation is taken
    from the exported dynamics instead -- the same constants the app uses, which were checked
    against MuJoCo to 5e-13 rad/s^2. The continuous Jacobian is finite-differenced and then
    discretised exactly with a zero-order hold.
    """
    C = EXP.fit_dynamics(env)
    C["v_eps"] = 0.005
    # Coulomb friction is not differentiable at zero speed, and the tanh stand-in turns it into a
    # huge fake viscous damping there (0.012 / 0.005 = 2.4 N m per rad/s), which wrecks the
    # linearisation. Design against the frictionless plant; friction only helps a balancer.
    accel = EXP.js_model(dict(C, fric_arm=0.0, fric_pend=0.0))
    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    def f(x, u):
        a = math.pi + x[1]
        A, B = accel(a, x[2], x[3], u * env.tau_max)
        return np.array([x[2], x[3], A, B])

    n, eps = 4, 1e-6
    Ac = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = eps
        Ac[:, j] = (f(x0 + dx, 0.0) - f(x0 - dx, 0.0)) / (2 * eps)
    Bc = ((f(x0, eps) - f(x0, -eps)) / (2 * eps)).reshape(n, 1)

    dt = env.ctrl_dt
    Maug = np.zeros((n + 1, n + 1))
    Maug[:n, :n] = Ac * dt
    Maug[:n, n:] = Bc * dt
    E = expm(Maug)
    A = E[:n, :n]
    B = E[:n, n:]

    # Bryson's rule: weight each state by 1/(largest value we are willing to see)^2, so the
    # controller stays inside the +-1 N m torque clamp instead of asking for gains it cannot use.
    Q = np.diag([q_th, q_phi, q_thd, q_ald])
    R = np.array([[r_u]])
    P = Q.copy()
    for _ in range(iters):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn
            break
        P = Pn
    BtP = B.T @ P
    K = np.linalg.solve(R + BtP @ B, BtP @ A)
    eig = np.linalg.eigvals(A - B @ K)
    return K[0], A, B, float(np.max(np.abs(eig)))


def rollout(env, K, Jp, mgl, kE, sign, phi_sw, ald_sw, th0, al0, record=False):
    e_up = 2.0 * mgl
    obs = env.set_state(th0, al0, 0.0, 0.0)
    n = env.max_steps
    tail = int(round(2.0 / env.ctrl_dt))
    total = 0.0
    upr = np.zeros(n)
    mode = np.zeros(n)
    log = []
    for i in range(n):
        al = float(env.data.qpos[1])
        th = float(env.data.qpos[0])
        thd = float(env.data.qvel[0])
        ald = float(env.data.qvel[1])
        phi = wrap(al - math.pi)
        if abs(phi) < phi_sw and abs(ald) < ald_sw:
            u = -float(K[0] * th + K[1] * phi + K[2] * thd + K[3] * ald)
            mode[i] = 1.0
        else:
            E = 0.5 * Jp * ald * ald + mgl * (1.0 - math.cos(al))
            u = sign * kE * (e_up - E) * math.copysign(1.0, ald * math.cos(al))
        u = max(-1.0, min(1.0, u))
        _, rw, term, trunc, info = env.step(np.array([u], dtype=np.float32))
        total += rw
        upr[i] = info["upright"]
        if record:
            log.append((env.data.time, math.degrees(phi), ald, math.degrees(th), thd, u, mode[i]))
        if term:
            break
    return total, float(upr.mean()), float(upr[-tail:].mean()), log


def score(env, K, Jp, mgl, kE, sign, phi_sw, ald_sw):
    rets, ups, settled = [], [], []
    for th0, al0 in EVAL_ICS:
        t, u, s, _ = rollout(env, K, Jp, mgl, kE, sign, phi_sw, ald_sw, th0, al0)
        rets.append(t)
        ups.append(u)
        settled.append(s)
    return (float(np.mean(settled)), float(np.mean(rets)), float(np.mean(ups)),
            float(np.mean([s > 0.9 for s in settled])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--armature", type=float, default=5.0e-3)
    p.add_argument("--out", default=os.path.join(HERE, "baseline.json"))
    a = p.parse_args()

    t0 = time.perf_counter()
    env = FurutaEnv(arm_armature=a.armature)
    env.reset(seed=0)
    Jp, mgl = plant_model(env)
    K, A, B, rho = lqr_gain(env)
    t_design = time.perf_counter() - t0
    print("pendulum inertia about hinge Jp = %.3e kg m2,  m*g*l = %.4f N m" % (Jp, mgl))
    print("LQR gain K = [%.3f %.3f %.3f %.3f]  (theta, phi, theta_dot, alpha_dot),  rho = %.4f"
          % (K[0], K[1], K[2], K[3], rho))
    print("design time (model Jacobian + Riccati) = %.3f s" % t_design)

    best = None
    for sign in (1.0, -1.0):
        for kE in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0):
            for phi_sw in (0.25, 0.4, 0.6):
                for ald_sw in (2.0, 4.0, 8.0):
                    s = score(env, K, Jp, mgl, kE, sign, phi_sw, ald_sw)
                    if best is None or s[0] > best[0][0]:
                        best = (s, dict(sign=sign, kE=kE, phi_sw=phi_sw, ald_sw=ald_sw))
    s, cfg = best
    print("best swing-up config %s" % cfg)
    print("settled=%.3f  ret=%.1f  upright=%.3f  success=%.2f" % s)

    _, _, _, log = rollout(env, K, Jp, mgl, cfg["kE"], cfg["sign"], cfg["phi_sw"],
                           cfg["ald_sw"], 0.0, 0.0, record=True)
    print("\n  t[s]   phi[deg]  ad[r/s]  theta[deg]   td[r/s]     u   mode")
    for row in log[::10]:
        print("%6.2f  %9.1f  %7.2f  %10.1f  %8.2f  %5.2f  %4.0f" % row)

    rec = dict(algo="lqr", kind="classical", seed=0, device="cpu", n_envs=0,
               setup_time=round(t_design, 3), train_time=0.0, steps=0,
               fps=0, budget=180.0, armature=a.armature, task="swingup",
               ctrl_hz=1.0 / env.ctrl_dt, timestep=float(env.model.opt.timestep),
               episode_steps=env.max_steps, tag="classical", K=[float(x) for x in K],
               swingup=cfg, Jp=Jp, mgl=mgl, rho=rho,
               points=[dict(t=0.0, steps=0, ret=s[1], ret_min=s[1], upright=s[2],
                            settled=s[0], success=s[3])])
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
