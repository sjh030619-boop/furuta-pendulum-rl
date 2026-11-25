"""진자 결합항(gam)이 CAD 값인지, 조원 B의 식별값(CAD x 0.8)인지 실기 기록으로 가릅니다.

    python s2r/coupling_fit.py

방법 (팔 운동 강제 재생)
    팔 각도는 실측 궤적 그대로 강제하고, 진자만 MuJoCo 의 진자 행 운동방정식으로 적분합니다.

        M10(α)·θ̈ + (M11(α) + armature)·α̈ + bias1(α, θ̇, α̇) = τ_pend(α̇)

    팔 마찰·관성·모터 모델의 오차는 여기 들어오지 않습니다(팔은 실측대로 움직이므로).
    진자 쪽 남는 미지수는 결합항 M10 하나이고, 이를 k 배 해서 진자 각도 오차가 가장 작은 k 를 찾습니다.
    M10 = gam·cos(α) 꼴이라 k 배 = 조원 B 의 gam_scale 과 같은 뜻입니다.

    진자 마찰은 조원 A 의 M2 모델(자유진동 실험 1·1b 로 맞춘 값)을 그대로 씁니다.
    진자가 도립 근처에 가면 아주 작은 오차도 커지므로 짧은 구간(0.6 s)마다 실측으로 다시 시작하고
    (multiple shooting), 구간 안에서 |α| 가 137° 를 넘는 구간은 뺍니다.

입력: s2r/pi_mirror_20261124/pi_host/logs/run_policy_2026112*_HHMMSS.csv (50 Hz, 각도는 시뮬 규약)
    theta 는 0xA1 응답 값이라 약 한 주기(19 ms) 전 값입니다. 시간 맞춤 d(샘플)도 같이 훑습니다.
"""
import os
import sys
import glob
import json
import math
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
MIR = os.path.join(HERE, "pi_mirror_20261124")
XML = os.path.join(os.path.dirname(HERE), "sim2real_rev11_xml", "furuta_rev11_bent_with_tip.xml")
for c in (XML, r"C:\Users\JH\Desktop\Furuta_rev11_bent_MuJoCo\furuta_rev11_bent_with_tip.xml"):
    if os.path.exists(c):
        XML = c
        break

PEND = json.load(open(os.path.join(MIR, "params", "pendulum.json"), encoding="utf-8"))["M2"]
SEG = 0.6
STRIDE = 0.2
ALPHA_MAX = math.radians(137.0)
H = 0.002


def pendulum_tables():
    """진자 행의 M10, M11, bias1 을 α 격자에서 미리 계산. bias1 은 속도의 2차식."""
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    m.dof_armature[:] = 0.0
    al = np.linspace(-math.pi, math.pi, 1441)
    M10 = np.zeros_like(al)
    M11 = np.zeros_like(al)
    g = np.zeros_like(al)
    ctt = np.zeros_like(al)
    cta = np.zeros_like(al)
    caa = np.zeros_like(al)
    full = np.zeros((2, 2))

    def b1(a, td, ad):
        d.qpos[:] = [0.0, a]
        d.qvel[:] = [td, ad]
        mujoco.mj_forward(m, d)
        return float(d.qfrc_bias[1])

    for i, a in enumerate(al):
        d.qpos[:] = [0.0, a]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, d, full)
        M10[i], M11[i] = full[1, 0], full[1, 1]
        g[i] = b1(a, 0.0, 0.0)
        ctt[i] = b1(a, 1.0, 0.0) - g[i]
        caa[i] = b1(a, 0.0, 1.0) - g[i]
        cta[i] = b1(a, 1.0, 1.0) - g[i] - ctt[i] - caa[i]
    return al, M10, M11, g, ctt, cta, caa


def load_runs():
    files = sorted(f for f in glob.glob(os.path.join(MIR, "pi_host", "logs", "run_policy_2026112*_*.csv"))
                   if os.path.basename(f).count("_") == 3)
    runs = []
    for f in files:
        a = np.genfromtxt(f, delimiter=",", names=True)
        if a.size < 100 or np.max(np.abs(a["u"])) < 1e-6:
            continue
        runs.append((os.path.basename(f)[11:26], a))
    return runs


def segments(runs, d):
    """구간마다 (시간축, θ̇, θ̈ (H/2 격자), α 실측, α0, α̇0)."""
    out = []
    n_sub = int(round(SEG / H))
    for name, a in runs:
        t = a["t"]
        th = a["theta"]
        al = np.unwrap(a["alpha"])
        if d:
            th = np.concatenate([th[d:], np.full(d, th[-1])])
        sth = CubicSpline(t, th)
        sal = CubicSpline(t, al)
        t0 = 0.0
        while t0 + SEG <= t[-1] - 0.1:
            tt = t0 + np.arange(2 * n_sub + 1) * (H / 2)
            alm = sal(tt[::2])
            wrapped = (alm + math.pi) % (2 * math.pi) - math.pi
            if np.max(np.abs(wrapped)) < ALPHA_MAX and np.std(sth(tt, 2)) > 2.0:
                out.append(dict(run=name, t0=t0, thd=sth(tt, 1), thdd=sth(tt, 2), al_meas=alm,
                                al0=float(sal(t0)), ad0=float(sal(t0, 1))))
            t0 += STRIDE
    return out


def simulate(segs, tab, k):
    """모든 구간을 한꺼번에 RK4 로. 반환: 구간별 진자각 궤적 (n_seg, n_sub+1)."""
    al_g, M10, M11, g, ctt, cta, caa = tab
    arm = PEND["armature"]
    fl, damp, c2 = PEND["frictionloss"], PEND["damping"], PEND["c2"]
    n = len(segs)
    n_sub = (segs[0]["thdd"].size - 1) // 2
    THD = np.stack([s["thd"] for s in segs])
    THDD = np.stack([s["thdd"] for s in segs])
    a = np.array([s["al0"] for s in segs])
    v = np.array([s["ad0"] for s in segs])
    traj = np.zeros((n, n_sub + 1))
    traj[:, 0] = a

    def interp(tbl, x):
        w = (x + math.pi) % (2 * math.pi) - math.pi
        return np.interp(w, al_g, tbl)

    def acc(a, v, j):
        td, tdd = THD[:, j], THDD[:, j]
        tau = -damp * v - fl * np.tanh(v / 0.05) - c2 * v * np.abs(v)
        b = interp(g, a) + interp(ctt, a) * td * td + interp(cta, a) * td * v + interp(caa, a) * v * v
        return (tau - k * interp(M10, a) * tdd - b) / (interp(M11, a) + arm)

    for i in range(n_sub):
        j0, j1, j2 = 2 * i, 2 * i + 1, 2 * i + 2
        k1a, k1v = v, acc(a, v, j0)
        k2a, k2v = v + 0.5 * H * k1v, acc(a + 0.5 * H * k1a, v + 0.5 * H * k1v, j1)
        k3a, k3v = v + 0.5 * H * k2v, acc(a + 0.5 * H * k2a, v + 0.5 * H * k2v, j1)
        k4a, k4v = v + H * k3v, acc(a + H * k3a, v + H * k3v, j2)
        a = a + H / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
        v = v + H / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        traj[:, i + 1] = a
    return traj


def main():
    tab = pendulum_tables()
    i0 = np.argmin(np.abs(tab[0]))
    print("XML:", XML)
    print("CAD 결합항 |M10(α=0)| = %.4e  (조원 B 의 CAD gam 2.0631e-04, 식별 1.6505e-04)" % abs(tab[1][i0]))
    runs = load_runs()
    print("실기 정책 실행 기록 %d개: %s ... %s" % (len(runs), runs[0][0], runs[-1][0]))
    ks = np.round(np.arange(0.50, 1.301, 0.05), 3)
    res = {}
    for d in (0, 1, 2):
        segs = segments(runs, d)
        meas = np.stack([s["al_meas"] for s in segs])
        rms = []
        for k in ks:
            tr = simulate(segs, tab, k)
            rms.append(float(np.degrees(np.sqrt(np.mean((tr - meas) ** 2)))))
        kb = float(ks[int(np.argmin(rms))])
        per_run = {}
        for name in sorted(set(s["run"] for s in segs)):
            idx = [i for i, s in enumerate(segs) if s["run"] == name]
            sub = [segs[i] for i in idx]
            r = [float(np.sqrt(np.mean((simulate(sub, tab, k) - meas[idx]) ** 2))) for k in ks]
            per_run[name] = float(ks[int(np.argmin(r))])
        res[d] = dict(k_best=kb, rms_deg=dict(zip([str(k) for k in ks], rms)), n_seg=len(segs),
                      per_run=per_run)
        pr = np.array(list(per_run.values()))
        print("\n시간 맞춤 d=%d 샘플 (%d ms), 구간 %d개" % (d, 20 * d, len(segs)))
        print("  k:   " + " ".join("%5.2f" % k for k in ks))
        print("  RMS: " + " ".join("%5.2f" % r for r in rms) + "  [deg]")
        print("  -> 전체 최적 k = %.2f  (k=1.0 RMS %.2f°, k=0.8 RMS %.2f°)"
              % (kb, rms[list(ks).index(1.0)], rms[list(ks).index(0.8)]))
        print("  -> 실행별 최적 k: 중앙값 %.2f, 범위 %.2f~%.2f (%d회)"
              % (np.median(pr), pr.min(), pr.max(), len(pr)))
    with open(os.path.join(HERE, "coupling_fit.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print("\nwrote s2r/coupling_fit.json")


if __name__ == "__main__":
    main()
