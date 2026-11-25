"""교육용 키트 안전 검토: 스윙업 중 사람에 닿았을 때의 접촉량.

    python rl/safety.py --out rl/safety.json

토크값만으로는 위험도를 말할 수 없습니다. 사람에게 전달되는 것은
  - 눌림/끼임: 정적 접촉력 = tau / r  (r = 접촉점의 회전축 거리)
  - 충돌: 접촉점 선속도와 **유효 질량**. 유효 질량에는 기어로 묶인 로터 환산 관성이
    그대로 들어갑니다 (J_eff / r^2). 기어비가 높은 액추에이터가 위험한 이유입니다.
  - 운동에너지: 0.5 v^T M v. M 은 armature 를 포함하므로 로터 몫이 들어 있습니다.

접촉점은 둘을 봅니다. 진자 봉 끝(회전축에서 가장 먼 점)과 진자 힌지 높이의 봉(r=0.08 m).
"""
import os
import sys
import json
import math
import argparse
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torque as T

CFG = dict(sign=-1.0, kE=40.0, phi_sw=0.25, ald_sw=4.0)
K_DEFAULT = [-0.370, 4.909, -0.174, 0.460]


def site_id(env, name):
    return mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, name)


def kinematics(env, sid):
    """사이트의 위치·선속도, 전체 운동에너지, 그리고 **반사 질량**.

    반사 질량(operational-space inertia)은 그 점을 그 방향으로 밀 때 실제로 느껴지는
    질량입니다. m_eff = 1 / (e^T J M^-1 J^T e). 진자 힌지가 자유롭기 때문에 봉을 치면
    로터 관성이 그대로 오지 않고 상당 부분이 진자 회전으로 빠집니다 — 그 몫까지 반영됩니다."""
    m, d = env.model, env.data
    pos = np.array(d.site_xpos[sid])
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, res, 0)
    vel = res[3:6]
    full = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, full)
    v = np.array(d.qvel)
    ke = 0.5 * float(v @ full @ v)
    r = float(math.hypot(pos[0], pos[1]))
    sp = float(np.linalg.norm(vel))

    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, sid)
    lam = jp @ np.linalg.inv(full) @ jp.T
    if sp > 1e-9:
        e = vel / sp
        q = float(e @ lam @ e)
        m_eff = 1.0 / q if q > 1e-12 else float("inf")
    else:
        m_eff = float("nan")
    return pos, vel, sp, ke, r, m_eff


def run(env, K, tau_max):
    """스윙업 롤아웃 전체에서 접촉 관련 최대치를 모읍니다."""
    sid = site_id(env, "tip")
    Jp, mgl = T.CL.plant_model(env)
    rows = []
    for ic in T.EVAL_ICS:
        env.set_state(ic[0], ic[1], 0.0, 0.0)
        for _i in range(env.max_steps):
            th = float(env.data.qpos[0])
            al = float(env.data.qpos[1])
            thd = float(env.data.qvel[0])
            ald = float(env.data.qvel[1])
            phi = T.CL.wrap(al - math.pi)
            if abs(phi) < CFG["phi_sw"] and abs(ald) < CFG["ald_sw"]:
                u = -float(K[0] * th + K[1] * phi + K[2] * thd + K[3] * ald)
                bal = 1.0
            else:
                E = 0.5 * Jp * ald * ald + mgl * (1.0 - math.cos(al))
                u = CFG["sign"] * CFG["kE"] * (2.0 * mgl - E) * math.copysign(1.0, ald * math.cos(al))
                bal = 0.0
            u = max(-1.0, min(1.0, u))
            env.step(np.array([u], dtype=np.float32))
            _p, _v, sp, ke, r, me = kinematics(env, sid)
            imp = 0.5 * me * sp * sp if np.isfinite(me) else 0.0
            rows.append((float(env.data.ctrl[0]), sp, ke, r, thd, ald, bal, me, imp))
    a = np.array(rows)
    sw = a[:, 6] < 0.5
    k = int(np.argmax(a[:, 8]))
    return dict(
        tau_max=tau_max,
        tip_speed_peak=float(a[:, 1].max()),
        tip_speed_peak_swing=float(a[sw, 1].max()),
        tip_speed_rms=float(np.sqrt(np.mean(a[:, 1] ** 2))),
        ke_peak=float(a[:, 2].max()),
        ke_peak_swing=float(a[sw, 2].max()),
        ke_rms=float(np.sqrt(np.mean(a[:, 2] ** 2))),
        r_max=float(a[:, 3].max()),
        r_at_ke_peak=float(a[int(np.argmax(a[:, 2])), 3]),
        w_arm_peak=float(np.abs(a[:, 4]).max()),
        w_pend_peak=float(np.abs(a[:, 5]).max()),
        tau_peak=float(np.abs(a[:, 0]).max()),
        m_eff_median=float(np.nanmedian(a[:, 7])),
        m_eff_max=float(np.nanmax(a[np.isfinite(a[:, 7]), 7])),
        impact_ke_peak=float(a[k, 8]),
        impact_at=dict(speed=float(a[k, 1]), m_eff=float(a[k, 7]), r=float(a[k, 3])),
    )


def contact(tau_max, J_total, r):
    """한 접촉 반경에서의 정적 접촉력과 유효 질량."""
    return dict(r=r, static_force=tau_max / r, eff_mass=J_total / (r * r))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(HERE, "safety.json"))
    a = p.parse_args()

    out = {}
    e0 = T.build(1.0, 16.0)
    inert = T.arm_inertia(e0)
    J_tot = inert["J_total_hang"]
    sid = site_id(e0, "tip")

    e0.set_state(0.0, math.pi * 0.5, 0.0, 0.0)
    mujoco.mj_forward(e0.model, e0.data)
    _p, _v, _s, _k, r_h, _m = kinematics(e0, sid)
    e0.set_state(0.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(e0.model, e0.data)
    _p, _v, _s, _k, r_d, _m = kinematics(e0, sid)
    out["geometry"] = dict(J_total=J_tot, J_mech=inert["J_mech_hang"],
                           r_tip_horizontal=r_h, r_tip_hanging=r_d,
                           pend_mass=float(e0.model.body_mass[2]),
                           arm_mass=float(e0.model.body_mass[1]))
    print("진자 봉 끝의 수직축 거리: 매달린 자세 %.3f m, 수평 자세 %.3f m (= 스윙 반경 최대)"
          % (r_d, r_h))
    print("팔 축 총 관성 %.3e kg m2 (기구부 %.3e + 로터환산 5.0e-3)"
          % (J_tot, inert["J_mech_hang"]))

    print("\n=== 토크 클램프별 접촉량 (스윙업 전 구간의 최대치) ===")
    print("%8s %9s %8s %8s %8s %10s %10s" %
          ("tau[Nm]", "팁[m/s]", "KE[J]", "w_arm", "w_pend", "반사질량kg", "충돌KE[J]"))
    runs = []
    for tm in (1.0, 0.6, 0.4):
        env = T.build(tm, 16.0)
        Jp, mgl = T.CL.plant_model(env)
        K, _A, _B, _r = T.CL.lqr_gain(env)
        d = run(env, [float(x) for x in K], tm)
        runs.append(d)
        print("%8.2f %9.2f %8.3f %8.2f %8.2f %10.4f %10.4f"
              % (tm, d["tip_speed_peak"], d["ke_peak"], d["w_arm_peak"],
                 d["w_pend_peak"], d["impact_at"]["m_eff"], d["impact_ke_peak"]))
    out["runs"] = runs

    print("\n=== 접촉 반경별 정적 접촉력 / 유효 질량 ===")
    print("%10s %14s %14s %14s" % ("r[m]", "힘@1.0Nm[N]", "힘@0.4Nm[N]", "유효질량[kg]"))
    cs = []
    for r in (0.05, 0.08, 0.12, r_h):
        c1 = contact(1.0, J_tot, r)
        c2 = contact(0.4, J_tot, r)
        cs.append(dict(r=r, f_1_0=c1["static_force"], f_0_4=c2["static_force"],
                       eff_mass=c1["eff_mass"]))
        print("%10.3f %14.1f %14.1f %14.3f"
              % (r, c1["static_force"], c2["static_force"], c1["eff_mass"]))
    out["contact"] = cs

    print("\n=== 충돌 등가 (반사 질량 기준, 진자 봉 끝 접촉) ===")
    for d in runs:
        i = d["impact_at"]
        print("  tau %.2f N m: %.2f m/s x 반사질량 %.4f kg -> 충돌 KE %.4f J"
              % (d["tau_max"], i["speed"], i["m_eff"], d["impact_ke_peak"]))
        print("      기계 전체 KE %.3f J / 반사질량 중앙값 %.4f kg, 최대 %.4f kg"
              % (d["ke_peak"], d["m_eff_median"], d["m_eff_max"]))
    out["J_total"] = J_tot

    print("\n=== 모터 능력 기준 정적 접촉력 [N] (클램프가 풀렸을 때) ===")
    print("%16s %9s %9s %9s" % ("토크[Nm]", "r=0.08", "r=0.12", "r=0.161"))
    for lab, tm in (("과제 요구 0.4", 0.4), ("현재 클램프 1.0", 1.0),
                    ("RMD-X6 정격 20", 20.0), ("RMD-X6 피크 60", 60.0)):
        print("%16s %9.0f %9.0f %9.0f" % (lab, tm / 0.08, tm / 0.12, tm / r_h))

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
