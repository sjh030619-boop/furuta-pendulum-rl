"""키트 맞추기 계산 (모터를 움직이지 않습니다. 기록을 받아 값만 계산).

kit_server.py 가 매 실행 수평 측정과 연결 보정에서 부릅니다. numpy 만 씁니다(Pi 에 scipy 없음).

  fit_level(stops, C=None)     정지 자리마다 (팔 0x92 각도°, 진자 RAW 중앙값) -> 영점 식 C + A·cos(θ − φ)
  fit_arm(rows, KT)            마찰 펄스 기록 (t, iq_cmd, iq_reply, speed_dps) -> 팔 관성 J, 쿨롱 Fc±, 점성 b
  fit_swing(ts, raws)          팔을 잠그고 진자가 흔들리는 기록 -> 작은 진폭 주기 T0, 감쇠
  model_period(model)          앱 모델(model.json)이 예측하는 진자 주기
"""
import math

import numpy as np

COUNTS = 4096.0


def counts_to_deg(c):
    return c * 360.0 / COUNTS


def fit_level(stops, C=None):
    """stops: [(a92_deg, raw), ...]. C 를 주면 A, φ 만 맞춥니다(정지 자리 2곳 이상), 안 주면 C 도(3곳 이상).
    돌려주는 것: C, A(카운트), phi_deg, tilt_deg(진자 영점이 팔 각도에 따라 바뀌는 폭 = 받침 기울기), resid(카운트 RMS)."""
    th = np.radians([s[0] for s in stops])
    y = np.array([float(s[1]) for s in stops])
    if C is None:
        M = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
        (c, a, b), *_ = np.linalg.lstsq(M, y, rcond=None)
    else:
        c = float(C)
        M = np.column_stack([np.cos(th), np.sin(th)])
        (a, b), *_ = np.linalg.lstsq(M, y - c, rcond=None)
    pred = c + a * np.cos(th) + b * np.sin(th)
    A = math.hypot(a, b)
    phi = math.degrees(math.atan2(b, a)) % 360.0
    resid = float(np.sqrt(np.mean((y - pred) ** 2)))
    return dict(C=float(c), A=float(A), phi_deg=float(phi), tilt_deg=counts_to_deg(A), resid=resid, n=len(stops))


def zero_fn(C, A, phi_deg):
    def z(a92_deg):
        if a92_deg is None:
            return C
        return C + A * math.cos(math.radians(a92_deg - phi_deg))
    return z


def _arm_sim(t, tau, w0, J, Fp, Fm, b, eps=0.05, h=0.001):
    """기록된 토크로 팔 속도를 다시 계산 (구간 [t_k, t_k+1) 동안 토크 일정)."""
    w = np.empty(len(t))
    w[0] = v = w0
    for k in range(len(t) - 1):
        n = max(1, int(round((t[k + 1] - t[k]) / h)))
        dt = (t[k + 1] - t[k]) / n
        for _ in range(n):
            fc = Fp if v > 0 else Fm
            v += dt * (tau[k] - fc * math.tanh(v / eps) - b * v) / J
        w[k + 1] = v
    return w


def _nelder_mead(f, x0, step, iters=160):
    n = len(x0)
    pts = [np.array(x0, dtype=float)]
    for i in range(n):
        p = np.array(x0, dtype=float)
        p[i] += step[i]
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2.0 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fcv = f(xc)
            if fcv < vals[-1]:
                pts[-1], vals[-1] = xc, fcv
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def fit_arm(rows, KT):
    """rows: [(t, iq_cmd, iq_reply, speed_dps), ...] (run_policy.measure_friction 의 frows 앞 네 열).
    토크 = 다음 응답 iq x KT (조원 sim/quick_id.py 와 같은 정렬). 1) 식 오차 최소제곱으로 시작값,
    2) 기록 토크로 팔 속도를 다시 계산해 속도 오차가 가장 작은 값으로 다듬음 (quick_id 의 재생 맞춤과 같은 목적)."""
    a = np.array(rows, dtype=float)
    t = a[:, 0]
    iq = a[:, 2]
    w = np.radians(a[:, 3])
    tau = np.append(iq[1:], iq[-1]) * 0.01 * KT
    ws = np.convolve(w, np.ones(3) / 3.0, mode="same")
    wd = np.gradient(ws, t)
    m = np.abs(ws) > 0.2
    M = np.column_stack([wd[m], (ws[m] > 0).astype(float), -(ws[m] < 0).astype(float), ws[m]])
    (J0, Fp0, Fm0, b0), *_ = np.linalg.lstsq(M, tau[m], rcond=None)
    J0 = min(max(J0, 0.005), 0.08)
    Fp0, Fm0, b0 = max(Fp0, 0.02), max(Fm0, 0.02), max(b0, 0.0)

    def cost(x):
        J, Fp, Fm, b = x
        if J <= 0.002 or Fp < 0 or Fm < 0 or b < 0:
            return 1e9
        return float(np.sqrt(np.mean((_arm_sim(t, tau, w[0], J, Fp, Fm, b) - w) ** 2)))

    x, rms = _nelder_mead(cost, [J0, Fp0, Fm0, b0], [0.2 * J0, 0.1, 0.1, 0.05])
    return dict(J=float(x[0]), Fc_pos=float(x[1]), Fc_neg=float(x[2]), viscous=float(x[3]),
                speed_rms_dps=math.degrees(rms), start=dict(J=float(J0), Fc_pos=float(Fp0), Fc_neg=float(Fm0), viscous=float(b0)))


def fit_swing(ts, raws, amp_lo_deg=3.0, amp_hi_deg=25.0):
    """팔을 잠근 채 흔들리는 진자. 같은 방향 영점 통과 사이 시간 = 한 주기, 주기마다 진폭으로 큰 진폭 효과를 고쳐
    작은 진폭 주기 T0 = T / (1 + θ0²/16) 로 바꿉니다. 진폭이 amp_lo~amp_hi 인 주기만 씁니다."""
    t = np.asarray(ts, dtype=float)
    r = np.asarray(raws, dtype=float)
    r = (r - r[0] + COUNTS / 2) % COUNTS - COUNTS / 2 + r[0]
    tail = r[int(len(r) * 0.8):]
    center = float(np.median(tail))
    x = np.radians(counts_to_deg(r - center))
    up = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
    cross = [t[i] + (t[i + 1] - t[i]) * (-x[i]) / (x[i + 1] - x[i]) for i in up]
    T0s, amps = [], []
    for k in range(len(cross) - 1):
        sel = (t >= cross[k]) & (t < cross[k + 1])
        if sel.sum() < 6:
            continue
        amp = 0.5 * (x[sel].max() - x[sel].min())
        if not math.radians(amp_lo_deg) <= amp <= math.radians(amp_hi_deg):
            continue
        T = cross[k + 1] - cross[k]
        T0s.append(T / (1.0 + amp * amp / 16.0))
        amps.append(math.degrees(amp))
    if not T0s:
        return dict(ok=False, n=0, center=center)
    T0s = np.array(T0s)
    decay = None
    if len(amps) >= 3:
        decay = float(-np.polyfit(np.arange(len(amps)), np.log(amps), 1)[0])
    return dict(ok=True, T0=float(np.median(T0s)), T0_std=float(np.std(T0s)), n=int(len(T0s)),
                amp_max=float(max(amps)), amp_min=float(min(amps)), decay_per_cycle=decay, center=center)


def model_period(model):
    """앱 모델의 진자(팔 고정) 작은 진폭 주기: 2π·sqrt(M11 / g2)."""
    return 2.0 * math.pi * math.sqrt(model["m11"] / model["g1"][2])
