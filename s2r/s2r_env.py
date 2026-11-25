"""rev11 실기 보정 통일 환경 (PPO·SAC 비교용).

두 조원의 결과를 하나로 합친 환경입니다. 값과 출처는 아래 표, 판정 근거는 s2r/README.md.

| 항목 | 값 | 출처 |
|---|---|---|
| 플랜트 | rev11 with_tip XML (CAD) | rev11_model |
| 팔 armature J | 공칭 0.0245 kg m^2, 무작위 0.0210~0.0260 (몸체 2.36e-4 는 별도) | 조원 A kit_identify replay(11-20, 11-23), exp2/2b. 11-23: A 0.0250+몸체 = B 총 0.0252 로 일치 |
| 팔 쿨롱 Fc+/Fc- | 공칭 0.69/0.68 N m, 무작위 평균 0.42~0.85, 비대칭 +-10% | 같음 (MuJoCo 재생 맞춤 값) |
| 팔 점성 b | 공칭 0.17 N m s/rad, 무작위 0.06~0.30 로그균등 | 같음 (온도 30~47 C 범위) |
| 진자 | 조원 A M2 (자유진동 맞춤) | params/pendulum.json |
| 결합항 | CAD x 0.87 (95% 0.86~0.89), 무작위 0.80~0.95 | s2r/coupling_fit_exp2.py |
| 토크 경로 | u*2.52 -> 속도-토크 제한(16 rad/s) -> 1.2 A 상한 -> 제동 5 W(직립 +-20 밖) -> 0.01 A 양자화 | pi_host/run_policy.py 와 같은 순서 |
| 관측 | [cos th, sin th, cos al, sin al, thd/20, ald/40], 50 Hz | run_policy.py 와 같음 (배포 시 호스트 수정 없음) |
| 관측 시점 | th, thd 는 직전 주기 응답(약 1 주기 전), al 은 최신, ald 는 20 ms 차분 | run_policy.py |
| 양자화 | th 0.01 deg, thd 1 dps, al 12비트 | 같음 |
| 명령 지연 | 관측 뒤 0~2 서브스텝(0~4 ms) 뒤 적용 | CAN 왕복 0.42 ms + 계산 |
| 실제/명령 토크 | 공칭 0.95, 무작위 0.90~1.00 | 실기 28회: 응답 전류 = 0.95 x 명령 (상관 1.00, s2r/README.md) |
| 진자 영점 오차 | 에피소드마다 +-1.5 deg | 영점 식 미끄러짐 관측(최대 약 2 deg) |
| 종료 | 팔 > 1100 dps 또는 진자 > 25 rad/s (호스트가 멈추는 조건), 벌점 -200 | run_policy.py |

마찰과 모터 토크는 매 물리 스텝 qfrc_applied 로 넣습니다(XML 모터의 +-1 N m 클램프와 MuJoCo
frictionloss 의 약한 소프트 제약을 피함. 두 조원 모두 같은 결론).
"""
import os
import sys
import math
import json
import numpy as np
import mujoco
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "rl"))
from furuta_env import FurutaEnv, TWO_PI  # noqa: E402

PARAMS = os.path.join(HERE, "pi_mirror_20261124", "params")


def _find_xml():
    name = "furuta_rev11_bent_with_tip.xml"
    for c in (os.environ.get("FURUTA_REV11_XML", ""),
              os.path.join(ROOT, "rev11_model", name),
              os.path.join(r"C:\Users\JH\Desktop\Furuta_rev11_bent_MuJoCo", name)):
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("rev11 XML 을 찾지 못했습니다 (FURUTA_REV11_XML)")


XML = _find_xml()
KT = 2.1
IQ_MAX = 120
TAU_MAX = IQ_MAX * 0.01 * KT
W_NO_LOAD = 16.0
BRAKE_W = 5.0
BRAKE_ZONE = math.radians(20.0)
ARM_SPEED_LIM = math.radians(1100.0)
PEND_SPEED_LIM = 25.0
TERM_PENALTY = 200.0
AS5600 = TWO_PI / 4096.0

NOMINAL = dict(J=0.0245, Fp=0.69, Fm=0.68, b=0.17, k=0.87, pend_fl=1.0, zero_deg=0.0, lat=1, kt=0.95)
RANGE = dict(J=(0.0210, 0.0260), Fc=(0.42, 0.85), asym=(0.90, 1.10), b=(0.06, 0.30),
             k=(0.80, 0.95), pend_fl=(0.5, 2.0), zero_deg=(-1.5, 1.5), lat=(0, 2), kt=(0.90, 1.00))


def _pendulum_m2():
    with open(os.path.join(PARAMS, "pendulum.json"), encoding="utf-8") as fh:
        return json.load(fh)["M2"]


def quantize(x, q):
    return np.round(x / q) * q


class S2REnv(FurutaEnv):
    """randomize: False = 공칭값 고정, True = 에피소드마다 RANGE 안에서 뽑기.
    params: 일부 값을 직접 고정 (예: dict(J=0.025, k=0.8)). randomize 와 같이 쓰면 그 값만 고정."""

    def __init__(self, randomize=False, params=None, episode_sec=8.0, init="mixed", du_pen=0.0,
                 w_smooth=0.002, w_effort=0.02, **kw):
        """du_pen: 보상에서 du_pen*(u-u_prev)^2 를 뺌 (4단계 떨림 대책, 기본 0 = 1단계와 같음).
        w_smooth, w_effort: 보상의 진자 속도·모터 힘 가중치 (앱 '따라 하기' 아홉 칸용, 기본값 = 1단계 보상)."""
        kw.setdefault("xml", XML)
        kw.setdefault("timestep", 0.002)
        super().__init__(arm_armature=0.0, tau_max=TAU_MAX, arm_no_load_speed=W_NO_LOAD,
                         episode_sec=episode_sec, **kw)
        m = self.model
        m.dof_frictionloss[self.arm_dof] = 0.0
        m.dof_damping[self.arm_dof] = 0.0
        pm = _pendulum_m2()
        self._pend = pm
        m.dof_damping[self.pend_dof] = pm["damping"]
        m.dof_frictionloss[self.pend_dof] = pm["frictionloss"]
        m.dof_armature[self.pend_dof] = pm["armature"]
        self._c2 = pm["c2"]
        self._pbody = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.body_iquat[self._pbody])
        R = R.reshape(3, 3)
        self._I_cad = R @ np.diag(m.body_inertia[self._pbody]) @ R.T
        self._dg = self._coupling_slope()
        self._randomize = randomize
        self._fixed = dict(params or {})
        self._init = init
        self.p = dict(NOMINAL, **self._fixed)
        self._apply(self.p)
        self._hist = []
        self._u_prev = 0.0
        self._du_pen = float(du_pen)
        self._ws = float(w_smooth)
        self._we = float(w_effort)

    def _M01(self):
        d = mujoco.MjData(self.model)
        d.qpos[:] = [0.0, math.pi]
        mujoco.mj_forward(self.model, d)
        M = np.zeros((2, 2))
        mujoco.mj_fullM(self.model, d, M)
        return abs(M[0, 1])

    def _set_I(self, I):
        w, V = np.linalg.eigh(I)
        if np.linalg.det(V) < 0:
            V[:, 0] = -V[:, 0]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, V.flatten())
        self.model.body_inertia[self._pbody] = w
        self.model.body_iquat[self._pbody] = q

    def _coupling_slope(self):
        """결합항 gam 이 Ixz 에 얼마나 민감한지 (조원 B xcheck._scale_gam 과 같은 방법)."""
        self._set_I(self._I_cad)
        g0 = self._M01()
        I2 = self._I_cad.copy()
        I2[0, 2] = I2[2, 0] = I2[0, 2] + 1e-6
        self._set_I(I2)
        g1 = self._M01()
        self._set_I(self._I_cad)
        self._gam_cad = g0
        return (g1 - g0) / 1e-6

    def set_coupling(self, k):
        I = self._I_cad.copy()
        I[0, 2] = I[2, 0] = I[0, 2] + (k - 1.0) * self._gam_cad / self._dg
        self._set_I(I)

    def _apply(self, p):
        m = self.model
        m.dof_armature[self.arm_dof] = p["J"]
        m.dof_frictionloss[self.pend_dof] = self._pend["frictionloss"] * p["pend_fl"]
        self.set_coupling(p["k"])

    def _J_body(self):
        """armature 0 일 때 팔 축 몸체 관성(진자 매달림). J 는 이것까지 포함한 팔 축 총 관성."""
        if not hasattr(self, "_jb"):
            arm = self.model.dof_armature[self.arm_dof]
            self.model.dof_armature[self.arm_dof] = 0.0
            d = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, d)
            M = np.zeros((2, 2))
            mujoco.mj_fullM(self.model, d, M)
            self._jb = float(M[0, 0])
            self.model.dof_armature[self.arm_dof] = arm
        return self._jb

    def sample(self):
        r = self.np_random
        R = RANGE
        fc = r.uniform(*R["Fc"])
        a = r.uniform(*R["asym"])
        p = dict(J=r.uniform(*R["J"]), Fp=fc * math.sqrt(a), Fm=fc / math.sqrt(a),
                 b=float(math.exp(r.uniform(math.log(R["b"][0]), math.log(R["b"][1])))),
                 k=r.uniform(*R["k"]), pend_fl=float(math.exp(r.uniform(math.log(0.5), math.log(2.0)))),
                 zero_deg=r.uniform(*R["zero_deg"]), lat=int(r.integers(R["lat"][0], R["lat"][1] + 1)),
                 kt=r.uniform(*R["kt"]))
        p.update(self._fixed)
        return p

    def arm_friction(self, w):
        p = self.p
        return -((p["Fp"] if w > 0 else p["Fm"]) * math.tanh(w / 0.05) + p["b"] * w)

    def _substeps(self, tau_old, tau_new, n_old):
        m, d = self.model, self.data
        a, pd = self.arm_dof, self.pend_dof
        qa, v = d.qfrc_applied, d.qvel
        c2 = self._c2
        for i in range(self.decim):
            qa[a] = (tau_old if i < n_old else tau_new) + self.arm_friction(v[a])
            vp = v[pd]
            qa[pd] = -c2 * vp * abs(vp)
            mujoco.mj_step(m, d)
            if i == 0:
                self._resp = (float(d.qpos[a]), float(d.qvel[a]))

    def _measure(self):
        """호스트가 이번 주기에 보는 값. th, thd 는 직전 주기 응답, al 은 지금 AS5600."""
        th, thd = self._resp
        al_true = float(self.data.qpos[1]) + math.radians(self.p["zero_deg"])
        al = quantize(((al_true + math.pi) % TWO_PI) - math.pi, AS5600)
        if self._al_prev is None:
            ald = 0.0
        else:
            dd = (al - self._al_prev + math.pi) % TWO_PI - math.pi
            ald = dd / self.ctrl_dt
        self._al_prev = al
        th = quantize(th, math.radians(0.01))
        thd = math.radians(round(math.degrees(thd)))
        self._meas = (th, thd, al, ald)
        return np.array([math.cos(th), math.sin(th), math.cos(al), math.sin(al),
                         thd / 20.0, ald / 40.0], dtype=np.float32)

    def command(self, u):
        """run_policy.py 와 같은 순서로 정책 출력 u -> 실제 토크."""
        th, thd, al, ald = self._meas
        tau = u * TAU_MAX
        if tau * thd > 0.0:
            lim = TAU_MAX * max(0.0, 1.0 - abs(thd) / W_NO_LOAD)
            tau = max(-lim, min(lim, tau))
        tau = max(-TAU_MAX, min(TAU_MAX, tau))
        phi = (al - math.pi + math.pi) % TWO_PI - math.pi
        if abs(phi) > BRAKE_ZONE and tau * thd < 0.0 and thd != 0.0:
            lim = BRAKE_W / abs(thd)
            tau = max(-lim, min(lim, tau))
        iq = int(round(tau / KT / 0.01))
        iq = max(-IQ_MAX, min(IQ_MAX, iq))
        return iq * 0.01 * KT

    def _start(self, th, al, thd, ald):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = [th, al]
        self.data.qvel[:] = [thd, ald]
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        self._resp = (th, thd)
        self._al_prev = None
        self._tau = 0.0
        self._u_prev = 0.0
        o = self._measure()
        self._al_prev = self._meas[2]
        return o

    def reset(self, *, seed=None, options=None):
        super(FurutaEnv, self).reset(seed=seed)
        self.p = self.sample() if self._randomize else dict(NOMINAL, **self._fixed)
        self._apply(self.p)
        r = self.np_random
        if self._init == "hang" or (self._init == "mixed" and r.random() < 0.6):
            th0, al0 = r.uniform(-0.2, 0.2), r.uniform(-0.5, 0.5)
            thd0, ald0 = 0.0, r.uniform(-0.5, 0.5)
        else:
            th0, al0 = r.uniform(-math.pi, math.pi), r.uniform(-math.pi, math.pi)
            thd0, ald0 = r.uniform(-3.0, 3.0), r.uniform(-6.0, 6.0)
        return self._start(th0, al0, thd0, ald0), {"params": dict(self.p)}

    def set_state(self, theta, alpha, theta_dot=0.0, alpha_dot=0.0):
        return self._start(theta, alpha, theta_dot, alpha_dot)

    def step(self, action):
        u = float(np.clip(action[0], -1.0, 1.0))
        tau_new = self.command(u) * self.p.get("kt", 1.0)
        self._substeps(self._tau, tau_new, int(self.p["lat"]))
        self._tau = tau_new
        q, v = self.data.qpos, self.data.qvel
        al, thd, ald = q[1], v[0], v[1]
        phi = (al - math.pi) % TWO_PI
        if phi > math.pi:
            phi -= TWO_PI
        near = math.exp(-11.111111 * phi * phi)
        slow = math.exp(-0.25 * ald * ald)
        r = (1.0 - 0.3 * phi * phi - self._ws * ald * ald - 0.002 * thd * thd
             - self._we * u * u + 0.5 * near * slow)
        if self._du_pen:
            r -= self._du_pen * (u - self._u_prev) ** 2
        self._u_prev = u
        self._step += 1
        terminated = bool(abs(thd) > ARM_SPEED_LIM or abs(ald) > PEND_SPEED_LIM)
        if terminated:
            r -= TERM_PENALTY
        truncated = self._step >= self.max_steps
        up = 1.0 if (-0.1745 < phi < 0.1745 and -2.0 < ald < 2.0) else 0.0
        info = {"upright": up, "phi": phi, "ald": ald, "tau": tau_new}
        return self._measure(), r, terminated, truncated, info


def make_env(**kw):
    def _f():
        return S2REnv(**kw)
    return _f
