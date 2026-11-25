"""rev11 보정 환경과 학습한 정책을 앱(app/src/assets)으로 내보냅니다.

    python s2r/export_app.py --model            # model.js, model.json + JS 검증용 기준 자료(app_ref.json)
    python s2r/export_app.py --policies         # policies.js (runs_app_rules 의 아홉 칸 + 바탕 정책)
    python s2r/export_app.py                    # 둘 다

1. 동역학 상수. 질량행렬은 진자 각도에만 의존하므로 삼각다항식 몇 개로 끝납니다(rl/export_for_app.py 와 같은 방법).
   결합항 배율 k 는 진자 관성의 Ixz 만 바꾸고 관성은 Ixz 에 선형이므로, k=0 과 k=1 에서 맞춘 계수를 선형 보간합니다.
   팔 armature J 는 계수에서 빼 두고 앱이 조건마다 더합니다.
2. 토크 경로·관측·양자화·지연·종료 조건의 상수 (s2r_env.py 와 같은 값).
3. 조건 섞기 5개와 안전 시험 8개 조건, 시작 자세. 앱의 안전 시험은 40칸입니다: 파이썬 빠른 평가의 시작 4개에
   '완전히 매달려 멈춘 상태' (0, 0) 를 더했습니다(실제 키트의 시작 상태). 37칸 이상이면 통과.
4. 기준 자료 app_ref.json: 같은 행동 열을 파이썬 환경에 넣었을 때의 궤적과, 바탕 정책의 고정 시험 결과.
   s2r/check_app_physics.js 가 앱의 JS 물리로 같은 것을 돌려 비교합니다.
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
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
ASSETS = os.path.join(ROOT, "app", "src", "assets")
RULES = os.path.join(HERE, "runs_app_rules")
KEYS = ["latent_pi.0.weight", "latent_pi.0.bias", "latent_pi.2.weight", "latent_pi.2.bias", "mu.weight", "mu.bias"]
ROBUST = [{}, dict(Fp=0.85, Fm=0.85, b=0.25), dict(Fp=0.45, Fm=0.45, b=0.08), dict(kt=0.90, lat=2), dict(k=0.80)]
QUICK = (0, 5, 10, 15)


def fit_at(env, k):
    """armature 0, 결합 배율 k 에서 M00(a), M01(a), M11, G1(a) 를 삼각다항식으로."""
    m = env.model
    arm = env.arm_dof
    keep = float(m.dof_armature[arm])
    m.dof_armature[arm] = 0.0
    env.set_coupling(k)
    d = mujoco.MjData(m)
    al = np.linspace(-math.pi, math.pi, 721)
    M00, M01, M11, G1 = (np.zeros_like(al) for _ in range(4))
    full = np.zeros((m.nv, m.nv))
    for i, a in enumerate(al):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = [0.0, a]
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, d, full)
        M00[i], M01[i], M11[i] = full[0, 0], full[0, 1], full[1, 1]
        G1[i] = d.qfrc_bias[1]
    m.dof_armature[arm] = keep

    def fit(y, basis):
        A = np.stack([f(al) for f in basis], axis=1)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return c, float(np.max(np.abs(A @ c - y)))

    one = lambda x: np.ones_like(x)  # noqa: E731
    c00, e00 = fit(M00, [one, np.cos, np.sin, lambda x: np.cos(2 * x), lambda x: np.sin(2 * x)])
    c01, e01 = fit(M01, [one, np.cos, np.sin])
    cg, eg = fit(G1, [one, np.cos, np.sin])
    return dict(m00=c00, m01=c01, m11=float(np.mean(M11)), g1=cg,
                err=dict(m00=e00, m01=e01, m11=float(np.max(np.abs(M11 - np.mean(M11)))), g1=eg))


def build_model():
    import s2r_env as E
    from evaluate import CONDITIONS, STARTS, EP_SEC, HOLD_SEC
    env = E.S2REnv(params=dict(E.NOMINAL))
    f0, f1, fh = fit_at(env, 0.0), fit_at(env, 1.0), fit_at(env, 0.5)
    env.set_coupling(E.NOMINAL["k"])
    lin = max(float(np.max(np.abs(fh[k] - 0.5 * (f0[k] + f1[k])))) for k in ("m00", "m01"))
    pm = env._pend
    C = dict(
        name="rev11", made_by="s2r/export_app.py",
        m00=[float(x) for x in f0["m00"]], m00_k=[float(x) for x in (f1["m00"] - f0["m00"])],
        m01=[float(x) for x in f0["m01"]], m01_k=[float(x) for x in (f1["m01"] - f0["m01"])],
        m11=f0["m11"], g1=[float(x) for x in f0["g1"]],
        pend_damping=float(pm["damping"]), pend_fl=float(pm["frictionloss"]), pend_c2=float(pm["c2"]), pend_fl_eps=0.02,
        arm_fric_eps=0.05,
        kt_amp=E.KT, iq_max=E.IQ_MAX, tau_max=E.TAU_MAX, arm_no_load_speed=E.W_NO_LOAD,
        brake_w=E.BRAKE_W, brake_zone=E.BRAKE_ZONE, arm_speed_lim=E.ARM_SPEED_LIM, pend_speed_lim=E.PEND_SPEED_LIM,
        as5600=E.AS5600, th_q=math.radians(0.01), thd_q=math.radians(1.0),
        timestep=float(env.model.opt.timestep), ctrl_hz=1.0 / env.ctrl_dt, obs_scale=dict(arm=20.0, pend=40.0),
        episode_sec=8.0, nominal=dict(E.NOMINAL), robust=ROBUST,
        gate=dict(conds=[dict(name=k, over=v) for k, v in CONDITIONS.items()],
                  starts=[[STARTS[i][0], STARTS[i][1]] for i in QUICK] + [[0.0, 0.0]],
                  all_starts=[[a, b] for a, b in STARTS], secs=EP_SEC, hold=HOLD_SEC, pass_min=37),
        fit_err=dict(m00=max(f0["err"]["m00"], f1["err"]["m00"]), m01=max(f0["err"]["m01"], f1["err"]["m01"]),
                     m11=f0["err"]["m11"], g1=f0["err"]["g1"], k_linear=lin),
    )
    return C, env


def reference(env_cls, nominal):
    """JS 물리 검증용: 정해진 행동 열에 대한 파이썬 환경의 궤적 + 바탕 정책의 고정 시험 결과."""
    from evaluate import run_episode, CONDITIONS, STARTS
    from policy_np import ActorNP
    out = dict(sequences=[], gates={})
    seqs = [
        ("sine", {}, [0.0, 0.0], lambda i: 0.6 * math.sin(2 * math.pi * 1.2 * i * 0.02)),
        ("pulses", {}, [0.0, 0.3], lambda i: (0.9 if (i // 15) % 2 == 0 else -0.9) if i < 90 else 0.0),
        ("sine_fric_hi", dict(Fp=0.85, Fm=0.85, b=0.25), [0.0, 0.0], lambda i: 0.8 * math.sin(2 * math.pi * 0.9 * i * 0.02)),
        ("sine_lat2", dict(kt=0.90, lat=2), [0.0, -0.2], lambda i: 0.7 * math.sin(2 * math.pi * 1.5 * i * 0.02)),
        ("sine_k080", dict(k=0.80, J=0.0260), [0.5, 0.1], lambda i: 0.7 * math.sin(2 * math.pi * 1.1 * i * 0.02)),
    ]
    for name, over, x0, fn in seqs:
        env = env_cls(params=dict(nominal, **over), episode_sec=4.0, init="hang")
        env.reset(seed=0)
        o = env.set_state(x0[0], x0[1], 0.0, 0.0)
        us, states, obs = [], [], [[float(v) for v in o]]
        for i in range(150):
            u = float(fn(i))
            o, _r, term, _tr, info = env.step(np.array([u], dtype=np.float32))
            us.append(u)
            states.append([float(env.data.qpos[0]), float(env.data.qpos[1]), float(env.data.qvel[0]), float(env.data.qvel[1]), float(info["tau"])])
            obs.append([float(v) for v in o])
            if term:
                break
        out["sequences"].append(dict(name=name, over=over, x0=x0, u=us, states=states, obs=obs))
    for f in sorted(glob.glob(os.path.join(HERE, "policies", "base_sac_s*.npz"))):
        a = ActorNP(f)

        def pol(o):
            return a(o.astype(np.float64))
        cells, holds = [], []
        for cname, over in CONDITIONS.items():
            env = env_cls(params=dict(nominal, **over), episode_sec=12.0, init="hang")
            env.reset(seed=0)
            for th, al in STARTS:
                r = run_episode(env, pol, th, al)
                cells.append(int(r["strict"]))
                holds.append(r["t_hold"])
        w = np.load(f)
        out["gates"][os.path.basename(f)[:-4]] = dict(
            cells=cells, total=int(sum(cells)), t_hold=holds,
            layers=[dict(w=w[KEYS[2 * i]].tolist(), b=w[KEYS[2 * i + 1]].tolist()) for i in range(3)])
    return out


def policy_spec(npz_path, name, curve, extra=None):
    w = np.load(npz_path)
    layers = [dict(w=[[round(float(x), 6) for x in row] for row in w[KEYS[2 * i]]],
                   b=[round(float(x), 6) for x in w[KEYS[2 * i + 1]]]) for i in range(3)]
    spec = dict(algo="sac", name=name, activation="relu", squash=True,
                obs=["cos_theta", "sin_theta", "cos_alpha", "sin_alpha", "theta_dot/20", "alpha_dot/40"],
                curve=curve, layers=layers)
    spec.update(extra or {})
    return spec


def app_gate(npz_path):
    """앱의 안전 시험 40칸(조건 8개 x 시작 5개)을 파이썬 환경으로 봅니다."""
    import s2r_env as E
    from evaluate import run_episode, CONDITIONS, STARTS
    from policy_np import ActorNP
    a = ActorNP(npz_path)
    starts = [STARTS[i] for i in QUICK] + [(0.0, 0.0)]
    n = 0
    for over in CONDITIONS.values():
        env = E.S2REnv(params=dict(E.NOMINAL, **over), episode_sec=12.0, init="hang")
        env.reset(seed=0)
        n += sum(run_episode(env, lambda o: a(o.astype(np.float64)), th, al)["strict"] for th, al in starts)
    return int(n)


def export_policies(out_dir):
    bank = {}
    for f in sorted(glob.glob(os.path.join(RULES, "sac_s*e*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        bank[r["name"]] = policy_spec(f[:-5] + ".npz", r["name"], r["curve"],
                                      dict(label=r["label"], gate=app_gate(f[:-5] + ".npz"), gate32=r["gate32"],
                                           full128=r["full_strict"], train_sec=r["t_best"], stopped=r["stopped"]))
    with open(os.path.join(out_dir, "policies.js"), "w", encoding="utf-8") as fh:
        fh.write("window.PS = window.PS || {};\nPS.POLICIES = ")
        json.dump(bank, fh, separators=(",", ":"), ensure_ascii=False)
        fh.write(";\n")
    return bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--policies", action="store_true")
    ap.add_argument("--out", default=ASSETS)
    a = ap.parse_args()
    both = not (a.model or a.policies)
    os.makedirs(a.out, exist_ok=True)
    if a.model or both:
        import s2r_env as E
        C, _env = build_model()
        print("맞춤 잔차:", {k: "%.2e" % v for k, v in C["fit_err"].items()})
        with open(os.path.join(a.out, "model.json"), "w", encoding="utf-8") as fh:
            json.dump(C, fh, indent=1)
        with open(os.path.join(a.out, "model.js"), "w", encoding="utf-8") as fh:
            fh.write("window.PS = window.PS || {};\nPS.MODEL = ")
            json.dump(C, fh, separators=(",", ":"))
            fh.write(";\n")
        ref = reference(E.S2REnv, E.NOMINAL)
        with open(os.path.join(HERE, "app_ref.json"), "w", encoding="utf-8") as fh:
            json.dump(ref, fh)
        print("model.js 저장, 기준 자료:", {k: v["total"] for k, v in ref["gates"].items()})
    if a.policies or both:
        bank = export_policies(a.out)
        print("policies.js 저장: %d개 (안전 시험 40칸, 128칸)" % len(bank), {k: (v["gate"], v["full128"]) for k, v in bank.items()})


if __name__ == "__main__":
    main()
