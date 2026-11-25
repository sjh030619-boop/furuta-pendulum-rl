"""구동 지연과 행동 이력을 다루는 환경.

`furuta_env.py` 는 건드리지 않고 상속합니다. 벤치마크가 쓰는 환경의 동작과 속도를 그대로
두기 위해서입니다. 보상·종료·관측의 앞 6칸은 부모와 같습니다.

지연
    부모의 `action_delay_steps` 는 제어 주기(20 ms)의 정수배만 됩니다. 실제 지연은
    그렇지 않으므로(USB 왕복 5~25 ms) 물리 서브스텝(2 ms) 단위로 줍니다.
    제어 스텝 k 에서 고른 명령 u_k 는 `delay_ms` 뒤부터 모터에 들어갑니다. 지연이
    d 서브스텝이면 full, frac = divmod(d, 10) 이고, 스텝 k 의 앞 frac 서브스텝에는
    u_{k-full-1}, 나머지에는 u_{k-full} 이 걸립니다. frac = 0 이면 부모의 정수 지연과
    완전히 같습니다(`python rl/delay_env.py` 가 그것을 확인합니다).

    delay_ms        고정 지연
    delay_range_ms  (lo, hi). 에피소드마다 그 안에서 균일하게 뽑습니다. 학습용.
    jitter_ms       스텝마다 0~jitter 를 더합니다. USB 처럼 매번 다른 지연을 흉내 냅니다.

행동 이력
    hist = K 이면 관측 뒤에 직전 명령 K개를 오래된 것부터 붙입니다.
    K = 2 일 때 [..., u_{k-2}, u_{k-1}] 로, 앱의 8차원 관측(`physics.js` 의 `_uh`)과
    순서까지 같습니다. 지연이 있으면 "보냈지만 아직 안 먹은 명령"이 숨은 상태가 되는데,
    이력을 관측에 넣으면 그것이 다시 보입니다.
"""
import os
import sys
import math
import numpy as np
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from furuta_env import FurutaEnv, TWO_PI


class DelayEnv(FurutaEnv):
    def __init__(self, delay_ms=0.0, delay_range_ms=None, jitter_ms=0.0, hist=0, **kw):
        super().__init__(**kw)
        self.sub_ms = float(self.model.opt.timestep) * 1000.0
        self.delay_ms = float(delay_ms)
        self.delay_range = tuple(delay_range_ms) if delay_range_ms else None
        self.jitter_sub = int(round(float(jitter_ms) / self.sub_ms))
        self.hist = int(hist)
        if self.hist:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6 + self.hist,),
                                                dtype=np.float32)
        self._d = int(round(self.delay_ms / self.sub_ms))
        self._cmds = [0.0] * 8

    def _begin(self, sample):
        if sample and self.delay_range is not None:
            lo = int(round(self.delay_range[0] / self.sub_ms))
            hi = int(round(self.delay_range[1] / self.sub_ms))
            self._d = int(self.np_random.integers(lo, hi + 1))
        elif self.delay_range is None:
            self._d = int(round(self.delay_ms / self.sub_ms))
        self._cmds = [0.0] * 8

    def _aug(self, o):
        if not self.hist:
            return o
        return np.concatenate([o, np.asarray(self._cmds[-self.hist:], dtype=np.float32)])

    def reset(self, *, seed=None, options=None):
        o, info = super().reset(seed=seed, options=options)
        self._begin(True)
        return self._aug(o), info

    def set_state(self, theta, alpha, theta_dot=0.0, alpha_dot=0.0):
        o = super().set_state(theta, alpha, theta_dot, alpha_dot)
        self._begin(False)
        return self._aug(o)

    def _tau(self, u, w):
        tau = u * self.tau_max
        if self.w0 > 0.0 and tau * w > 0.0:
            lim = self.tau_max * (1.0 - abs(w) / self.w0)
            if lim < 0.0:
                lim = 0.0
            if tau > lim:
                tau = lim
            elif tau < -lim:
                tau = -lim
        return tau

    def step(self, action):
        u = float(action[0])
        if u < -1.0:
            u = -1.0
        elif u > 1.0:
            u = 1.0
        d = self._d
        if self.jitter_sub:
            d += int(self.np_random.integers(0, self.jitter_sub + 1))
        full, frac = divmod(d, self.decim)
        c = self._cmds
        c.append(u)
        if len(c) > 64:
            del c[:32]
        a_new = c[-1 - full]
        a_old = c[-2 - full]

        q = self._qpos
        v = self._qvel
        n = self.decim
        if frac:
            self._ctrl[0] = self._tau(a_old, v[0])
            self._mj_step(self.model, self.data, nstep=frac)
            self._ctrl[0] = self._tau(a_new, v[0])
            self._mj_step(self.model, self.data, nstep=n - frac)
            eff = (frac * a_old * a_old + (n - frac) * a_new * a_new) / n
        else:
            self._ctrl[0] = self._tau(a_new, v[0])
            self._mj_step(self.model, self.data, nstep=n)
            eff = a_new * a_new

        al = q[1]
        thd = v[0]
        ald = v[1]
        phi = (al - math.pi) % TWO_PI
        if phi > math.pi:
            phi -= TWO_PI
        near = math.exp(-11.111111 * phi * phi)
        slow = math.exp(-0.25 * ald * ald)
        r = (1.0 - 0.3 * phi * phi - 0.002 * ald * ald - 0.002 * thd * thd
             - 0.02 * eff + 0.5 * near * slow)

        self._step += 1
        terminated = bool(thd > self._lim_arm or thd < -self._lim_arm
                          or ald > self._lim_pend or ald < -self._lim_pend)
        truncated = self._step >= self.max_steps
        up = 1.0 if (-0.1745 < phi < 0.1745 and -2.0 < ald < 2.0) else 0.0
        return self._aug(self._obs()), r, terminated, truncated, {"upright": up}


def make_delay_env(**kw):
    def _f():
        return DelayEnv(**kw)
    return _f


def _check():
    """정수 지연에서 부모 환경과 한 비트도 다르지 않은지 확인합니다."""
    rng = np.random.default_rng(0)
    acts = rng.uniform(-1.0, 1.0, size=(250, 1)).astype(np.float32)
    for steps in (0, 1, 2):
        a = FurutaEnv(action_delay_steps=steps)
        b = DelayEnv(delay_ms=20.0 * steps)
        oa = a.set_state(0.1, 0.3, 0.0, 0.5)
        ob = b.set_state(0.1, 0.3, 0.0, 0.5)
        worst_o = float(np.max(np.abs(oa - ob)))
        worst_r = 0.0
        for u in acts:
            oa, ra, ta, _tr, _ia = a.step(u)
            ob, rb, tb, _tr, _ib = b.step(u)
            worst_o = max(worst_o, float(np.max(np.abs(oa - ob))))
            worst_r = max(worst_r, abs(ra - rb))
            if ta or tb:
                break
        print("지연 %d 스텝: 관측 차이 최대 %.1e, 보상 차이 최대 %.1e" % (steps, worst_o, worst_r))

    e = DelayEnv(delay_ms=10.0, hist=2)
    o = e.set_state(0.0, 0.0)
    print("이력 2개 관측 차원 %d, 시작 시 이력 %s" % (o.shape[0], o[6:]))
    for i, u in enumerate((0.5, -0.25, 1.0)):
        o, _r, _t, _tr, _i = e.step(np.array([u], dtype=np.float32))
        print("  명령 %+.2f 뒤 이력 [u(k-2), u(k-1)] = %s" % (u, o[6:]))

    print("지연이 커질수록 같은 명령열의 궤적이 연속적으로 변하는지:")
    ref = None
    for ms in (0, 2, 6, 10, 14, 18, 20):
        e = DelayEnv(delay_ms=float(ms))
        e.set_state(0.0, 0.0)
        for u in acts[:40]:
            e.step(u)
        al = float(e.data.qpos[1])
        ref = al if ref is None else ref
        print("  %2d ms: 40스텝 뒤 진자각 %+.4f rad (0 ms 대비 %+.4f)" % (ms, al, al - ref))


if __name__ == "__main__":
    _check()
