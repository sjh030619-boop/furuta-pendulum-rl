"""모터 선정 근거: 실제로 요구되는 토크와 속도를 재고, 한계를 낮춰가며 어디서 과제가 깨지는지 봅니다.

    python rl/torque.py --out rl/torque.json

모델의 구동기는 `<motor joint="arm" gear="1" ctrlrange="-1 1"/>` 이므로 제어 입력이 곧
팔 관절의 토크[N m] 입니다. 환경은 거기에 1차 토크-속도 곡선을 씌웁니다 —
구동 방향에서는 tau <= tau_stall (1 - |w|/w_noload), 제동 방향에서는 tau_stall 전부.

세 가지를 잽니다.

  1. 설계점(tau_stall 1.0 N m, w_noload 16 rad/s)에서 **실제로 쓰는** 토크와 속도.
     피크, RMS, 포화 시간 비율을 스윙업 구간과 밸런스 구간으로 나눠서.
  2. tau_stall 을 낮춰가며 **어디서 과제가 깨지는지**. 한계를 바꿀 때마다 제어기를
     처음부터 다시 설계합니다(LQR 은 모델에서 다시 계산, 스윙업 이득은 다시 훑기).
     낮은 한계에 맞춰 조율된 제어기로 재야 "그 모터로 되느냐"의 답이 됩니다.
  3. 세울 일 없이 **버티기만** 할 때의 요구치. 연속 정격 쪽 근거입니다.

토크와 속도는 맞물려 있습니다(속도가 낮으면 에너지를 못 넣어 토크가 더 필요해짐).
그래서 2번은 tau x w 격자로 훑습니다.
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

from furuta_env import FurutaEnv
from bench import EVAL_ICS
import classical as CL

PI = math.pi

BAL_ICS = [
    (0.0, PI, 0.0, 1.0),
    (0.0, PI + 0.15, 0.0, -1.0),
    (0.0, PI - 0.15, 0.0, 1.0),
    (0.15, PI + 0.08, 0.0, -1.0),
    (-0.15, PI - 0.08, 0.0, 1.0),
]

KE_GRID = (5.0, 10.0, 20.0, 40.0, 80.0)
PHI_GRID = (0.25, 0.4, 0.6)
ALD_GRID = (2.0, 4.0, 8.0)
SIGN = -1.0

TAU_FINE = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3,
            0.25, 0.22, 0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08, 0.06)
TAU_GRID = (1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.16, 0.12, 0.1, 0.08, 0.06)
W_GRID = (8.0, 12.0, 16.0, 24.0, 1.0e6)


def build(tau_max, w0, armature=5.0e-3, task="swingup"):
    env = FurutaEnv(arm_armature=armature, tau_max=tau_max,
                    arm_no_load_speed=w0, task=task)
    env.reset(seed=0)
    hi = max(1.0, tau_max) * 1.001
    env.model.actuator_ctrlrange[0] = [-hi, hi]
    return env


def arm_inertia(env):
    """팔 축에서 본 관성. 로터 환산 관성(armature)이 기구부보다 큰지 보려고 나눠서 봅니다."""
    import mujoco
    m, d = env.model, env.data
    out = {}
    for tag, al in (("hang", 0.0), ("up", PI)):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = [0.0, al]
        mujoco.mj_forward(m, d)
        full = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, full)
        tot = float(full[0, 0])
        out["J_total_" + tag] = tot
        out["J_mech_" + tag] = tot - float(m.dof_armature[env.arm_dof])
    return out


def design(env):
    """이 토크 한계에 맞춰 제어기를 새로 설계합니다. LQR 은 모델에서, 스윙업 이득은 훑어서."""
    Jp, mgl = CL.plant_model(env)
    K, _A, _B, rho = CL.lqr_gain(env)
    best = None
    for kE in KE_GRID:
        for phi_sw in PHI_GRID:
            for ald_sw in ALD_GRID:
                s = CL.score(env, K, Jp, mgl, kE, SIGN, phi_sw, ald_sw)
                if best is None or s[0] > best[0][0]:
                    best = (s, dict(sign=SIGN, kE=kE, phi_sw=phi_sw, ald_sw=ald_sw))
    s, cfg = best
    return dict(K=[float(x) for x in K], Jp=Jp, mgl=mgl, rho=rho, cfg=cfg,
                settled=s[0], ret=s[1], upright=s[2], success=s[3])


def trace(env, K, Jp, mgl, cfg, ic):
    """한 번 굴리며 적용된 토크와 각속도를 매 스텝 기록합니다."""
    th0, al0 = ic[0], ic[1]
    td0 = ic[2] if len(ic) > 2 else 0.0
    ad0 = ic[3] if len(ic) > 3 else 0.0
    e_up = 2.0 * mgl
    env.set_state(th0, al0, td0, ad0)
    kE, sign, phi_sw, ald_sw = cfg["kE"], cfg["sign"], cfg["phi_sw"], cfg["ald_sw"]
    rows = []
    for i in range(env.max_steps):
        th = float(env.data.qpos[0])
        al = float(env.data.qpos[1])
        thd = float(env.data.qvel[0])
        ald = float(env.data.qvel[1])
        phi = CL.wrap(al - PI)
        bal = abs(phi) < phi_sw and abs(ald) < ald_sw
        if bal:
            u = -float(K[0] * th + K[1] * phi + K[2] * thd + K[3] * ald)
        else:
            E = 0.5 * Jp * ald * ald + mgl * (1.0 - math.cos(al))
            u = sign * kE * (e_up - E) * math.copysign(1.0, ald * math.cos(al))
        u = max(-1.0, min(1.0, u))
        _o, _r, term, _tr, info = env.step(np.array([u], dtype=np.float32))
        tau = float(env.data.ctrl[0])
        rows.append((i * env.ctrl_dt, u, tau, thd, ald, phi, 1.0 if bal else 0.0,
                     info["upright"]))
        if term:
            break
    return np.array(rows, dtype=float)


def stats(env, d, ics):
    """피크는 초기조건 전체의 최대, RMS 는 전체를 모아서."""
    tr = [trace(env, d["K"], d["Jp"], d["mgl"], d["cfg"], ic) for ic in ics]
    tau = np.concatenate([t[:, 2] for t in tr])
    w = np.concatenate([t[:, 3] for t in tr])
    wp = np.concatenate([t[:, 4] for t in tr])
    bal = np.concatenate([t[:, 6] for t in tr]) > 0.5
    pw = tau * w

    def rms(x):
        return float(np.sqrt(np.mean(x * x))) if x.size else 0.0

    tmax = env.tau_max
    t_up = []
    for t in tr:
        k = np.flatnonzero(t[:, 6] > 0.5)
        t_up.append(float(t[k[0], 0]) if k.size else float("nan"))
    return dict(
        tau_peak=float(np.max(np.abs(tau))),
        tau_rms=rms(tau), tau_rms_swing=rms(tau[~bal]), tau_rms_bal=rms(tau[bal]),
        tau_peak_bal=float(np.max(np.abs(tau[bal]))) if bal.any() else 0.0,
        sat_duty=float(np.mean(np.abs(tau) > 0.95 * tmax)),
        sat_duty_swing=float(np.mean(np.abs(tau[~bal]) > 0.95 * tmax)) if (~bal).any() else 0.0,
        w_peak=float(np.max(np.abs(w))), w_rms=rms(w),
        wpend_peak=float(np.max(np.abs(wp))),
        p_peak=float(np.max(np.abs(pw))), p_rms=rms(pw),
        swing_frac=float(np.mean(~bal)),
        t_upright=float(np.nanmean(t_up)),
        envelope=[[round(float(a), 4), round(float(b), 4)]
                  for a, b in zip(tau[::2], w[::2])],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--armature", type=float, default=5.0e-3)
    p.add_argument("--out", default=os.path.join(HERE, "torque.json"))
    p.add_argument("--skip-grid", action="store_true")
    a = p.parse_args()

    t0 = time.perf_counter()
    out = dict(armature=a.armature)

    e = build(1.0, 16.0, a.armature)
    out["plant"] = dict(
        tau_max_design=1.0, w0_design=16.0,
        ctrl_hz=1.0 / e.ctrl_dt, timestep=float(e.model.opt.timestep),
        arm_frictionloss=float(e.model.dof_frictionloss[e.arm_dof]),
        pend_frictionloss=float(e.model.dof_frictionloss[e.pend_dof]),
        arm_damping=float(e.model.dof_damping[e.arm_dof]),
        armature=float(e.model.dof_armature[e.arm_dof]),
    )
    Jp, mgl = CL.plant_model(e)
    out["plant"]["Jp"] = Jp
    out["plant"]["mgl"] = mgl
    out["plant"].update(arm_inertia(e))
    print("팔 축 관성: 기구부 %.3e + 로터환산(armature) %.3e = %.3e kg m2  (로터가 %.1f배)"
          % (out["plant"]["J_mech_hang"], out["plant"]["armature"],
             out["plant"]["J_total_hang"],
             out["plant"]["armature"] / out["plant"]["J_mech_hang"]))

    print("=== 1. 설계점에서 실제로 쓰는 토크/속도 (tau_stall 1.0 N m, w_noload 16 rad/s) ===")
    d1 = design(e)
    print("  설계: K=%s kE=%.0f success=%.2f ret=%.1f"
          % ([round(x, 3) for x in d1["K"]], d1["cfg"]["kE"], d1["success"], d1["ret"]))
    s_sw = stats(e, d1, EVAL_ICS)
    out["design_point"] = dict(design=d1, swingup=s_sw)
    print("  스윙업+밸런스: tau_peak=%.3f  rms=%.3f (스윙업 %.3f / 밸런스 %.3f)  포화 %.0f%%"
          % (s_sw["tau_peak"], s_sw["tau_rms"], s_sw["tau_rms_swing"],
             s_sw["tau_rms_bal"], 100 * s_sw["sat_duty"]))
    print("  속도: w_peak=%.2f rad/s (%.0f rpm)  w_rms=%.2f   피크출력 %.2f W  rms %.2f W"
          % (s_sw["w_peak"], s_sw["w_peak"] * 60 / (2 * PI), s_sw["w_rms"],
             s_sw["p_peak"], s_sw["p_rms"]))
    print("  세우는 데 걸린 시간 %.2f s, 스윙업 구간이 전체의 %.0f%%"
          % (s_sw["t_upright"], 100 * s_sw["swing_frac"]))

    eb = build(1.0, 16.0, a.armature)
    db = dict(d1)
    s_bal = stats(eb, db, BAL_ICS)
    out["design_point"]["balance"] = s_bal
    print("  버티기만: tau_peak=%.3f  rms=%.3f  w_peak=%.2f rad/s  포화 %.0f%%"
          % (s_bal["tau_peak"], s_bal["tau_rms"], s_bal["w_peak"],
             100 * s_bal["sat_duty"]))

    print("\n=== 2. 토크 한계를 낮춰가며 (w_noload 16 rad/s, 한계마다 제어기 재설계) ===")
    print("  %8s %8s %8s %9s %8s %8s" % ("tau[Nm]", "success", "settled", "ret", "kE", "w_pk"))
    fine = []
    for tm in TAU_FINE:
        env = build(tm, 16.0, a.armature)
        d = design(env)
        s = stats(env, d, EVAL_ICS)
        fine.append(dict(tau_max=tm, w0=16.0, success=d["success"], settled=d["settled"],
                         ret=d["ret"], kE=d["cfg"]["kE"], tau_peak=s["tau_peak"],
                         sat_duty=s["sat_duty"], w_peak=s["w_peak"], p_peak=s["p_peak"],
                         t_upright=s["t_upright"]))
        print("  %8.2f %8.2f %8.3f %9.1f %8.0f %8.2f"
              % (tm, d["success"], d["settled"], d["ret"], d["cfg"]["kE"], s["w_peak"]))
    out["tau_sweep"] = fine
    ok = [r["tau_max"] for r in fine if r["success"] >= 1.0]
    out["tau_min_swingup"] = min(ok) if ok else None
    print("  >> 5/5 성공을 유지하는 최저 토크: %s N m"
          % (("%.2f" % min(ok)) if ok else "없음"))

    print("\n=== 3. 버티기만 할 때의 최저 토크 ===")
    print("  %8s %8s %8s %9s" % ("tau[Nm]", "success", "settled", "tau_pk"))
    bal = []
    for tm in TAU_FINE:
        env = build(tm, 16.0, a.armature)
        Jp2, mgl2 = CL.plant_model(env)
        K2, _A, _B, _r = CL.lqr_gain(env)
        d = dict(K=[float(x) for x in K2], Jp=Jp2, mgl=mgl2,
                 cfg=dict(sign=SIGN, kE=40.0, phi_sw=10.0, ald_sw=1e9))
        tr = [trace(env, d["K"], Jp2, mgl2, d["cfg"], ic) for ic in BAL_ICS]
        tail = int(round(2.0 / env.ctrl_dt))
        settled = float(np.mean([t[-tail:, 7].mean() for t in tr]))
        succ = float(np.mean([t[-tail:, 7].mean() > 0.9 for t in tr]))
        tau = np.concatenate([t[:, 2] for t in tr])
        w = np.concatenate([t[:, 3] for t in tr])
        bal.append(dict(tau_max=tm, success=succ, settled=settled,
                        tau_peak=float(np.max(np.abs(tau))),
                        tau_rms=float(np.sqrt(np.mean(tau * tau))),
                        w_peak=float(np.max(np.abs(w)))))
        print("  %8.2f %8.2f %8.3f %9.3f" % (tm, succ, settled, bal[-1]["tau_peak"]))
    out["balance_sweep"] = bal
    okb = [r["tau_max"] for r in bal if r["success"] >= 1.0]
    out["tau_min_balance"] = min(okb) if okb else None
    print("  >> 버티기만 할 때 최저 토크: %s N m"
          % (("%.2f" % min(okb)) if okb else "없음"))

    if not a.skip_grid:
        print("\n=== 4. 토크 x 무부하속도 격자 (성공률 5개 초기조건) ===")
        grid = []
        hdr = "  %8s" % "tau\\w0"
        for w in W_GRID:
            hdr += " %8s" % (("%.0f" % w) if w < 1e5 else "inf")
        print(hdr)
        for tm in TAU_GRID:
            row = "  %8.2f" % tm
            for w in W_GRID:
                env = build(tm, w, a.armature)
                d = design(env)
                grid.append(dict(tau_max=tm, w0=w, success=d["success"],
                                 settled=d["settled"], ret=d["ret"], kE=d["cfg"]["kE"]))
                row += " %8.2f" % d["success"]
            print(row)
        out["grid"] = grid

    out["elapsed_min"] = round((time.perf_counter() - t0) / 60.0, 2)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print("\nwrote %s  (%.1f min)" % (a.out, out["elapsed_min"]))


if __name__ == "__main__":
    main()
