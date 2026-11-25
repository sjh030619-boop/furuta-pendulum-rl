"""기어비 선정 근거: 로터 환산 관성이 요구 토크를 어떻게 바꾸는지.

    python rl/torque_gear.py --out rl/torque_gear.json

팔 축은 수직이라 정적 유지 토크가 0입니다. 그래서 요구 토크는 전부 **관성을 돌리는 데**
쓰이고, 팔 축에서 본 관성은 기구부 + 로터 환산(armature = N^2 J_rotor) 입니다.

기어비 N 을 올리면 출력 토크는 N 배로 늘지만 환산 관성은 N^2 배로 늘어납니다.
그래서 요구 토크도 같이 올라가고, 어느 지점을 넘으면 손해입니다. 그 지점을 재는 스크립트입니다.

환산 관성을 여러 값으로 두고, 각각에서 **과제가 되는 최저 토크 한계**를 찾습니다.
한계를 바꿀 때마다 제어기(LQR + 스윙업 이득)를 처음부터 다시 설계합니다.
"""
import os
import sys
import json
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torque as T

ARM_GRID = (0.0, 5.0e-4, 1.0e-3, 2.5e-3, 5.0e-3, 1.0e-2, 2.0e-2, 4.0e-2)
TAU_SCAN = (1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.22, 0.2, 0.18,
            0.16, 0.14, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03)


def tau_min(armature, w0=16.0, verbose=True):
    """내림차순으로 훑어 5/5 성공이 깨지는 직전 값을 돌려줍니다."""
    best = None
    rows = []
    miss = 0
    for tm in TAU_SCAN:
        env = T.build(tm, w0, armature)
        d = T.design(env)
        s = T.stats(env, d, T.EVAL_ICS)
        rows.append(dict(tau_max=tm, success=d["success"], settled=d["settled"],
                         ret=d["ret"], tau_peak=s["tau_peak"], w_peak=s["w_peak"],
                         p_peak=s["p_peak"], kE=d["cfg"]["kE"]))
        if verbose:
            print("    tau=%5.2f  success=%.2f settled=%.3f  tau_pk=%.3f w_pk=%5.2f"
                  % (tm, d["success"], d["settled"], s["tau_peak"], s["w_peak"]))
        if d["success"] >= 1.0:
            best = tm
            miss = 0
        elif best is not None:
            miss += 1
            if miss >= 2:
                break
    return best, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(HERE, "torque_gear.json"))
    p.add_argument("--w0", type=float, default=16.0)
    a = p.parse_args()

    t0 = time.perf_counter()
    e = T.build(1.0, a.w0, 5.0e-3)
    inert = T.arm_inertia(e)
    j_mech = inert["J_mech_hang"]
    print("기구부만의 팔 축 관성 J_mech = %.3e kg m2 (매달린 자세)" % j_mech)
    print("설계값 armature = 5.0e-3 kg m2 -> 로터가 기구부의 %.1f배\n" % (5.0e-3 / j_mech))

    out = dict(w0=a.w0, J_mech=j_mech, rows=[])
    print("%10s %10s %9s %9s %9s" % ("armature", "J_tot", "tau_min", "tau_pk@min", "w_pk@min"))
    for am in ARM_GRID:
        print("  armature = %.1e" % am)
        tm, rows = tau_min(am, a.w0)
        at = next((r for r in rows if r["tau_max"] == tm), None)
        out["rows"].append(dict(armature=am, J_total=am + j_mech, tau_min=tm, scan=rows))
        print("%10.1e %10.3e %9s %9s %9s" % (
            am, am + j_mech, ("%.2f" % tm) if tm else "none",
            ("%.3f" % at["tau_peak"]) if at else "-",
            ("%.2f" % at["w_peak"]) if at else "-"))

    ok = [(r["J_total"], r["tau_min"]) for r in out["rows"] if r["tau_min"]]
    if len(ok) >= 2:
        J = np.array([x[0] for x in ok])
        tm = np.array([x[1] for x in ok])
        sl, ic = np.polyfit(J, tm, 1)
        out["fit_linear"] = dict(slope=float(sl), intercept=float(ic))
        print("\n요구 토크 ~ %.2f * J_total + %.4f   (기울기 = 필요한 각가속도 %.0f rad/s^2)"
              % (sl, ic, sl))
        out["alpha_required"] = float(sl)
        out["tau_static_floor"] = float(ic)

    out["elapsed_min"] = round((time.perf_counter() - t0) / 60.0, 2)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print("\nwrote %s  (%.1f min)" % (a.out, out["elapsed_min"]))


if __name__ == "__main__":
    main()
