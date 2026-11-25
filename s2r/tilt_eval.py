"""받침 기울기 시험: 중력을 기울인 MuJoCo 에서 정책을 돌려, 영점 식 보정이 있을 때와 없을 때를 비교합니다.

  python s2r/tilt_eval.py <정책.npz> [기울기 목록 deg] [none|fix|both]
  결과(2026-11-25, 가운데 칸 sac_s1e1): app/kit/README.md 의 "받침 기울기가 정책에 주는 영향"

진짜 직립 = 그 팔 각도에서 진자가 매달리는 방향의 반대. 판정은 진짜 직립 기준(evaluate.py 와 같은 엄격 기준).
보정 없음: 정책이 센서 값(팔 기준 진자 각도) 그대로를 봄 = 받침이 평평하다고 가정한 영점.
보정 있음: 그 팔 각도의 매달림 방향을 빼고 봄 = 매 실행 수평 측정이 맞았을 때.
"""
import os
import sys
import math
import time
import numpy as np
import mujoco

S2R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S2R)
from s2r_env import S2REnv, NOMINAL, TWO_PI  # noqa: E402
import evaluate as E  # noqa: E402
from validate_closedloop import ActorNP  # noqa: E402


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


class TiltEnv(S2REnv):
    def __init__(self, tilt_deg=0.0, dir_deg=0.0, fix=False, fix_err_deg=0.0, **kw):
        super().__init__(**kw)
        b, d = math.radians(tilt_deg), math.radians(dir_deg)
        g = 9.81
        self.model.opt.gravity[:] = [g * math.sin(b) * math.cos(d), g * math.sin(b) * math.sin(d), -g * math.cos(b)]
        self.fix = fix
        self.fix_err = math.radians(fix_err_deg)
        self._grid = np.linspace(-math.pi, math.pi, 361)
        self._delta = np.array([self._hang(t) for t in self._grid])

    def _pot(self, th, al, dd):
        dd.qpos[:] = [th, al]
        mujoco.mj_kinematics(self.model, dd)
        mujoco.mj_comPos(self.model, dd)
        return -float(np.dot(self.model.opt.gravity, dd.xipos[self._pbody]))

    def _hang(self, th):
        dd = mujoco.MjData(self.model)
        lo, hi = -0.3, 0.3
        gr = (math.sqrt(5) - 1) / 2
        for _ in range(60):
            a = hi - gr * (hi - lo)
            b = lo + gr * (hi - lo)
            if self._pot(th, a, dd) < self._pot(th, b, dd):
                hi = b
            else:
                lo = a
        return 0.5 * (lo + hi)

    def delta(self, th):
        return float(np.interp(wrap(th), self._grid, self._delta))

    def _measure(self):
        th_true = float(self.data.qpos[0])
        corr = (self.delta(th_true) + self.fix_err) if self.fix else 0.0
        q = self.data.qpos
        saved = float(q[1])
        q[1] = saved - corr
        try:
            return super()._measure()
        finally:
            q[1] = saved

    def step(self, action):
        o, r, term, trunc, info = super().step(action)
        q = self.data.qpos
        info["phi"] = wrap(float(q[1]) - self.delta(float(q[0])) - math.pi)
        return o, r, term, trunc, info

    def _start(self, th, al, thd, ald):
        return super()._start(th, al + self.delta(th), thd, ald)


def run(pol, tilt, fix, dirs=(0.0, 90.0, 180.0, 270.0), err=0.0):
    tot = n = 0
    holds = []
    term = 0
    for d in dirs:
        for name, over in E.CONDITIONS.items():
            env = TiltEnv(tilt_deg=tilt, dir_deg=d, fix=fix, fix_err_deg=err,
                          params=dict(NOMINAL, **over), episode_sec=E.EP_SEC, init="hang")
            env.reset(seed=0)
            for th, al in E.STARTS:
                r = E.run_episode(env, pol, th, al)
                tot += r["strict"]
                n += 1
                term += r["terminated"]
                if r["t_hold"] is not None:
                    holds.append(r["t_hold"])
    return tot, n, (float(np.median(holds)) if holds else None), term


def main():
    path = sys.argv[1]
    tilts = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2,3,5").split(",")]
    mode = sys.argv[3] if len(sys.argv) > 3 else "both"
    actor = ActorNP(path)
    pol = lambda o: actor(o.astype(np.float64))  # noqa: E731
    env = TiltEnv(tilt_deg=2.4, dir_deg=0.0)
    amp = max(abs(x) for x in env._delta)
    print("확인: 기울기 2.4° 일 때 매달림 방향 변화 폭 %.3f° (팔 각도에 따라 ±)" % math.degrees(amp))
    print("정책", os.path.basename(path))
    for t in tilts:
        for fix in ((False, True) if mode == "both" else ((mode == "fix"),)):
            if fix and t == 0.0:
                continue
            t0 = time.time()
            s, n, h, term = run(pol, t, fix)
            print("  기울기 %4.1f°  %-6s 엄격 통과 %3d/%d (%.0f%%)  직립 시작 중앙 %s  넘어져 멈춤 %d  [%.0f s]" % (
                t, "보정" if fix else "보정없음", s, n, 100.0 * s / n, ("%.2f s" % h) if h else "-", term, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
