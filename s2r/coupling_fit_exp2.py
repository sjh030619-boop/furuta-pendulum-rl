"""결합항 판정, 촘촘한 기록 판: 팔 펄스 실험(exp2, exp2b)으로.

    python s2r/coupling_fit_exp2.py

coupling_fit.py 는 50 Hz 정책 실행 기록을 써서 팔 가속도 복원이 거칠었습니다(스윙업 중 토크가
20 ms 안에 뒤집힘). exp2 는 팔 각도 100 Hz(0x92, 0.01°), 진자 RAW 약 1.9 kHz 라 훨씬 촘촘하고,
진자가 매달린 근처에서 흔들려(±17°, exp2b 는 ±105°) 결합항 효과가 깨끗하게 보입니다.

같은 방법(팔 운동 강제, 진자 행만 적분, 짧은 구간마다 실측으로 다시 시작)이고, 진자 각도는
그날의 영점 식(11-18, 받침 2.5° 기울어진 상태: conventions_sensor_zero.json 의 '이력')으로 바꿉니다.
"""
import os
import sys
import csv
import json
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coupling_fit as CF

MIR = CF.MIR
ZERO = json.load(open(os.path.join(MIR, "params", "conventions_sensor_zero.json"),
                      encoding="utf-8"))["진자영점_RAW"]["이력"]
RECS = ["exp2_arm_steps_20261118_174103", "exp2b_highspeed_20261118_184300"]


def raw0(a92):
    return ZERO["C"] + ZERO["A"] * np.cos(np.radians(np.mod(a92, 360.0) - ZERO["phi_deg"]))


def load(rec):
    cyc_t, cyc_a = [], []
    fast_t, fast_r = [], []
    for r in csv.DictReader(open(os.path.join(MIR, "pi_host", "logs", rec + ".csv"))):
        if r["kind"] == "cyc" and r["a92"]:
            cyc_t.append(float(r["t"]))
            cyc_a.append(float(r["a92"]))
        elif r["kind"] == "fast" and r["raw"]:
            fast_t.append(float(r["t"]))
            fast_r.append(float(r["raw"]))
    ct, ca = np.array(cyc_t), np.array(cyc_a)
    ft, fr = np.array(fast_t), np.array(fast_r)
    o = np.argsort(ct)
    ct, ca = ct[o], ca[o]
    keep = np.concatenate([[True], np.diff(ct) > 1e-6])
    ct, ca = ct[keep], ca[keep]
    a92_at_fast = np.interp(ft, ct, ca)
    al = (fr - raw0(a92_at_fast)) * 2 * math.pi / 4096.0
    tg = np.arange(max(ct[0], ft[0]), min(ct[-1], ft[-1]), 0.002)
    al_g = np.interp(tg, ft, al)
    al_g = savgol_filter(al_g, 11, 3)
    th = np.radians(ca - ca[0])
    return ct, th, tg, al_g


def segments(rec_data, seg=0.6, stride=0.2):
    out = []
    n_sub = int(round(seg / CF.H))
    for name, (ct, th, tg, al) in rec_data:
        sth = CubicSpline(ct, th)
        sal = CubicSpline(tg, al)
        t0 = max(ct[0], tg[0]) + 0.05
        t_end = min(ct[-1], tg[-1]) - 0.05
        while t0 + seg <= t_end:
            tt = t0 + np.arange(2 * n_sub + 1) * (CF.H / 2)
            thdd = sth(tt, 2)
            alm = sal(tt[::2])
            if np.std(thdd) > 5.0 and np.ptp(alm) > math.radians(3.0) \
                    and np.max(np.abs(alm)) < CF.ALPHA_MAX:
                out.append(dict(run=name, t0=t0, thd=sth(tt, 1), thdd=thdd, al_meas=alm,
                                al0=float(sal(t0)), ad0=float(sal(t0, 1))))
            t0 += stride
    return out


def main():
    tab = CF.pendulum_tables()
    data = []
    for rec in RECS:
        d = load(rec)
        data.append((rec, d))
        print("%s: 팔 %d샘플(%.0f Hz), 진자 격자 %d점, 진자 범위 %.1f°~%.1f°" % (
            rec, d[0].size, 1 / np.median(np.diff(d[0])), d[2].size,
            math.degrees(d[3].min()), math.degrees(d[3].max())))
    ks = np.round(np.arange(0.50, 1.301, 0.05), 3)
    res = {}
    for name in RECS + ["both"]:
        sub = data if name == "both" else [x for x in data if x[0] == name]
        segs = segments(sub)
        meas = np.stack([s["al_meas"] for s in segs])
        rms = [float(np.degrees(np.sqrt(np.mean((CF.simulate(segs, tab, k) - meas) ** 2)))) for k in ks]
        r0 = float(np.degrees(np.sqrt(np.mean((CF.simulate(segs, tab, 0.0) - meas) ** 2))))
        kb = float(ks[int(np.argmin(rms))])
        res[name] = dict(n_seg=len(segs), k_best=kb, rms=dict(zip(map(str, ks), rms)), rms_k0=r0)
        print("\n%s: 구간 %d개" % (name, len(segs)))
        print("  k:   " + " ".join("%5.2f" % k for k in ks))
        print("  RMS: " + " ".join("%5.2f" % r for r in rms) + "  [deg]")
        print("  -> 최적 k = %.2f | RMS k=1.0 %.2f°, k=0.8 %.2f°, 결합 없음(k=0) %.2f°"
              % (kb, rms[list(ks).index(1.0)], rms[list(ks).index(0.8)], r0))
    with open(os.path.join(HERE, "coupling_fit_exp2.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
