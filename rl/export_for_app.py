"""Export everything the Electron app needs out of the validated MuJoCo model.

1. Dynamics constants. The Furuta mass matrix depends only on alpha, so the whole model is a
   handful of trigonometric coefficients. They are fitted against MuJoCo's own mj_fullM and
   qfrc_bias, then the resulting model is integrated side by side with MuJoCo and the maximum
   deviation is reported -- the app must not quietly drift away from the simulator the policies
   were trained in.

2. Trained policies. SB3 actors are plain MLPs; the weights go out as JSON so the app can run
   them with two matrix products and no ML runtime.

    python rl/export_for_app.py --models rl/models --out app/src/assets
"""
import os
import sys
import json
import math
import glob
import argparse
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from furuta_env import FurutaEnv, W_ARM, W_PEND, DEFAULTS


def fit_dynamics(env):
    """M00(a), M01(a), M11, G1(a) from MuJoCo, as trig polynomials."""
    m, d = env.model, env.data
    arm, pend = env.arm_dof, env.pend_dof
    a_arm = float(m.dof_armature[arm])
    a_pend = float(m.dof_armature[pend])

    al = np.linspace(-math.pi, math.pi, 721)
    M00 = np.zeros_like(al)
    M01 = np.zeros_like(al)
    M11 = np.zeros_like(al)
    G1 = np.zeros_like(al)
    full = np.zeros((m.nv, m.nv))
    for i, a in enumerate(al):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = [0.0, a]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, d, full)
        M00[i], M01[i], M11[i] = full[0, 0], full[0, 1], full[1, 1]
        G1[i] = d.qfrc_bias[1]

    def fit(y, basis):
        A = np.stack([f(al) for f in basis], axis=1)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return c, float(np.max(np.abs(A @ c - y)))

    one = lambda x: np.ones_like(x)
    b_m00 = [one, np.cos, np.sin, lambda x: np.cos(2 * x), lambda x: np.sin(2 * x)]
    b_m01 = [one, np.cos, np.sin]
    b_g = [one, np.cos, np.sin]
    c00, e00 = fit(M00, b_m00)
    c01, e01 = fit(M01, b_m01)
    cg, eg = fit(G1, b_g)
    c11 = float(np.mean(M11))
    e11 = float(np.max(np.abs(M11 - c11)))

    # MuJoCo's qM already carries dof_armature on the diagonal, so the fitted M00/M11 include it.
    # The app must not add it a second time.
    return dict(
        m00=[float(x) for x in c00], m01=[float(x) for x in c01],
        m11=c11, g1=[float(x) for x in cg],
        armature_arm=0.0, armature_pend=0.0,
        armature_arm_actual=a_arm, armature_pend_actual=a_pend,
        damp_arm=float(m.dof_damping[arm]), damp_pend=float(m.dof_damping[pend]),
        fric_arm=float(m.dof_frictionloss[arm]), fric_pend=float(m.dof_frictionloss[pend]),
        fit_err=dict(m00=e00, m01=e01, m11=e11, g1=eg),
    )


def js_model(C):
    """The same equations the app will run, in numpy, so we can check them against MuJoCo."""
    c00, c01, cg = C["m00"], C["m01"], C["g1"]

    def M(a):
        m00 = c00[0] + c00[1] * math.cos(a) + c00[2] * math.sin(a) \
            + c00[3] * math.cos(2 * a) + c00[4] * math.sin(2 * a) + C["armature_arm"]
        m01 = c01[0] + c01[1] * math.cos(a) + c01[2] * math.sin(a)
        m11 = C["m11"] + C["armature_pend"]
        return m00, m01, m11

    def dM(a):
        d00 = -c00[1] * math.sin(a) + c00[2] * math.cos(a) \
            - 2 * c00[3] * math.sin(2 * a) + 2 * c00[4] * math.cos(2 * a)
        d01 = -c01[1] * math.sin(a) + c01[2] * math.cos(a)
        return d00, d01

    def grav(a):
        return cg[0] + cg[1] * math.cos(a) + cg[2] * math.sin(a)

    def accel(a, thd, ald, tau):
        m00, m01, m11 = M(a)
        d00, d01 = dM(a)
        h0 = d00 * thd * ald + d01 * ald * ald
        h1 = -0.5 * d00 * thd * thd + grav(a)
        f0 = C["damp_arm"] * thd + C["fric_arm"] * math.tanh(thd / C["v_eps"])
        f1 = C["damp_pend"] * ald + C["fric_pend"] * math.tanh(ald / C["v_eps"])
        b0 = tau - h0 - f0
        b1 = -h1 - f1
        det = m00 * m11 - m01 * m01
        return (b0 * m11 - b1 * m01) / det, (b1 * m00 - b0 * m01) / det

    return accel


def rollout_js(C, dt, n, torque, x0):
    accel = js_model(C)
    th, a, thd, ald = x0
    out = []
    for i in range(n):
        tau = torque(i * dt)

        def f(s):
            A, B = accel(s[1], s[2], s[3], tau)
            return np.array([s[2], s[3], A, B])

        s = np.array([th, a, thd, ald])
        k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        th, a, thd, ald = s
        out.append((th, a, thd, ald))
    return np.array(out)


def rollout_mj(env, dt, n, torque, x0):
    m, d = env.model, env.data
    old = m.opt.timestep
    m.opt.timestep = dt
    mujoco.mj_resetData(m, d)
    d.qpos[:] = [x0[0], x0[1]]
    d.qvel[:] = [x0[2], x0[3]]
    mujoco.mj_forward(m, d)
    out = []
    for i in range(n):
        d.ctrl[0] = torque(i * dt)
        mujoco.mj_step(m, d)
        out.append((d.qpos[0], d.qpos[1], d.qvel[0], d.qvel[1]))
    m.opt.timestep = old
    return np.array(out)


def verify_accel(env, C, n=4000, seed=0):
    """The decisive check: the same state and torque must give the same acceleration.
    Coulomb friction is a solver constraint in MuJoCo and a smooth tanh here, so it is switched
    off for this comparison; its effect is measured separately by the trajectory check."""
    m, d = env.model, env.data
    fa, fp = float(m.dof_frictionloss[0]), float(m.dof_frictionloss[1])
    da, dp = float(m.dof_damping[0]), float(m.dof_damping[1])
    m.dof_frictionloss[:] = 0.0
    m.dof_damping[:] = 0.0
    C2 = dict(C, v_eps=0.005, fric_arm=0.0, fric_pend=0.0, damp_arm=0.0, damp_pend=0.0)
    accel = js_model(C2)
    rng = np.random.default_rng(seed)
    worst = 0.0
    scale = 0.0
    for _ in range(n):
        al = rng.uniform(-math.pi, math.pi)
        thd = rng.uniform(-16, 16)
        ald = rng.uniform(-30, 30)
        u = rng.uniform(-1, 1)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = [rng.uniform(-math.pi, math.pi), al]
        d.qvel[:] = [thd, ald]
        d.ctrl[0] = u
        mujoco.mj_forward(m, d)
        a_mj = np.array(d.qacc)
        a_js = np.array(accel(al, thd, ald, u * env.tau_max))
        worst = max(worst, float(np.max(np.abs(a_mj - a_js))))
        scale = max(scale, float(np.max(np.abs(a_mj))))
    m.dof_frictionloss[:] = [fa, fp]
    m.dof_damping[:] = [da, dp]
    return worst, scale


def verify(env, C, dt=0.002, secs=2.0):
    n = int(secs / dt)
    cases = [
        ("free fall from 170 deg", lambda t: 0.0, (0.0, math.radians(170.0), 0.0, 0.0)),
        ("torque pulse", lambda t: 0.4 if 0.5 <= t < 0.8 else 0.0, (0.0, 0.05, 0.0, 0.0)),
        ("resonant drive", lambda t: 0.5 * math.sin(2 * math.pi * 1.63 * t), (0.0, 0.02, 0.0, 0.0)),
    ]
    rows = []
    for name, tq, x0 in cases:
        A = rollout_js(C, dt, n, tq, x0)
        B = rollout_mj(env, dt, n, tq, x0)
        da = np.max(np.abs(np.unwrap(A[:, 1]) - np.unwrap(B[:, 1])))
        dth = np.max(np.abs(np.unwrap(A[:, 0]) - np.unwrap(B[:, 0])))
        rows.append((name, math.degrees(da), math.degrees(dth)))
    return rows


def find_curve(base, runs_dirs):
    """The reward curve this policy actually produced while training, for the app to replay."""
    for d in runs_dirs:
        f = os.path.join(d, base + ".json")
        if os.path.exists(f):
            r = json.load(open(f))
            return [round(p["ret"], 2) for p in r["points"]]
    return []


def export_policies(models_dir, out_dir, runs_dirs):
    import torch
    from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C
    from sb3_contrib import TQC, CrossQ
    CLS = dict(ppo=PPO, a2c=A2C, sac=SAC, td3=TD3, ddpg=DDPG, tqc=TQC, crossq=CrossQ)
    os.makedirs(out_dir, exist_ok=True)
    bank = {}
    for f in sorted(glob.glob(os.path.join(models_dir, "*.zip"))):
        base = os.path.basename(f)[:-4]
        algo = base.split("_")[0]
        if algo not in CLS:
            continue
        model = CLS[algo].load(f, device="cpu")
        pol = model.policy
        if algo in ("ppo", "a2c"):
            mods = list(pol.mlp_extractor.policy_net) + [pol.action_net]
            squash = False
        elif hasattr(pol.actor, "latent_pi"):
            mods = list(pol.actor.latent_pi) + [pol.actor.mu]
            squash = True
        else:
            mods = list(pol.actor.mu)
            squash = True
        seq, acts, bad = [], set(), None
        for mod in mods:
            if isinstance(mod, torch.nn.Linear):
                seq.append(mod)
            elif isinstance(mod, torch.nn.ReLU):
                acts.add("relu")
            elif isinstance(mod, torch.nn.Tanh):
                acts.add("tanh")
            elif isinstance(mod, torch.nn.Flatten):
                pass
            else:
                bad = type(mod).__name__
        if mods and isinstance(mods[-1], torch.nn.Tanh):
            squash = True
            acts.discard("tanh") if len(acts) > 1 else None
        if bad:
            print("  skip %s: actor contains %s, which this exporter cannot represent" % (base, bad))
            continue
        act = "tanh" if acts == {"tanh"} else "relu"
        layers = []
        for l in seq:
            layers.append(dict(w=[[round(x, 6) for x in row] for row in
                                  l.weight.detach().cpu().numpy().tolist()],
                               b=[round(x, 6) for x in l.bias.detach().cpu().numpy().tolist()]))
        bank[base] = dict(algo=algo, name=base, activation=act, squash=squash,
                          obs=(["cos_theta", "sin_theta", "cos_alpha", "sin_alpha",
                                "theta_dot/%g" % W_ARM, "alpha_dot/%g" % W_PEND]
                               + (["u[t-1]", "u[t]"] if layers[0]["w"][0].__len__() == 8 else [])),
                          curve=find_curve(base, runs_dirs), layers=layers)
    with open(os.path.join(out_dir, "policies.js"), "w", encoding="utf-8") as fh:
        fh.write("window.PS = window.PS || {};\nPS.POLICIES = ")
        json.dump(bank, fh, separators=(",", ":"))
        fh.write(";\n")
    return list(bank)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=os.path.join(HERE, "models"))
    p.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "app", "src", "assets"))
    p.add_argument("--armature", type=float, default=5.0e-3)
    a = p.parse_args()

    env = FurutaEnv(arm_armature=a.armature)
    C = fit_dynamics(env)
    C["v_eps"] = 0.005
    C["ctrl_hz"] = 1.0 / env.ctrl_dt
    C["timestep"] = float(env.model.opt.timestep)
    C["tau_max"] = env.tau_max
    C["arm_no_load_speed"] = env.w0
    C["obs_scale"] = dict(arm=W_ARM, pend=W_PEND)
    C["episode_sec"] = DEFAULTS["episode_sec"]

    print("fit residuals (absolute):")
    for k, v in C["fit_err"].items():
        print("  %-4s %.3e" % (k, v))
    worst, scale = verify_accel(env, C)
    print("\nacceleration vs MuJoCo over 4000 random states (friction off):")
    print("  worst %.3e rad/s^2   (largest acceleration seen %.1f rad/s^2)" % (worst, scale))
    print("\ntrajectory vs MuJoCo, 2 s, dt 2 ms, friction on (max angle deviation):")
    for name, da, dth in verify(env, C):
        print("  %-24s pendulum %8.4f deg   arm %8.4f deg" % (name, da, dth))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "model.json"), "w") as f:
        json.dump(C, f, indent=1)
    with open(os.path.join(a.out, "model.js"), "w", encoding="utf-8") as f:
        f.write("window.PS = window.PS || {};\nPS.MODEL = ")
        json.dump(C, f, separators=(",", ":"))
        f.write(";\n")
    print("\nwrote", os.path.join(a.out, "model.js"))

    if os.path.isdir(a.models):
        runs = [os.path.join(HERE, d) for d in ("runs_rules", "runs", "runs_b")]
        made = export_policies(a.models, a.out, runs)
        print("exported %d policies: %s" % (len(made), ", ".join(made)))
    else:
        print("no models directory at", a.models, "- skipping policy export")


if __name__ == "__main__":
    main()
