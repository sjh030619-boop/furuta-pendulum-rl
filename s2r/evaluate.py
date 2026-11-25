"""PPO·SAC 비교용 고정 평가. 모든 후보를 이 함수 하나로 잽니다.

    python s2r/evaluate.py            # 기준선 두 개(원래 정책 단독 / + 브리지 보상)를 잼

조건 8개 x 시작 16개 = 128 칸 (조원 B 의 평가와 같은 크기).
    nom      공칭 (J 0.0245, Fc 0.69/0.68, b 0.17, 결합 0.87, 토크 0.95, 지연 1 서브스텝)
    J_lo/J_hi            armature 0.0210 / 0.0260
    fric_lo/fric_hi      Fc 0.45, b 0.08 (따뜻할 때 쪽) / Fc 0.85, b 0.25 (차가울 때 쪽)
    k_lo/k_hi            결합 0.80 (조원 B) / 0.95
    kt_lat               토크 0.90 + 명령 지연 2 서브스텝
시작 16개: 매달린 근처. 진자 -30..30°, 팔 0 또는 1 rad, 정지.

판정 (두 가지를 같이 적고, 고르는 건 엄격 기준으로):
    엄격  12 s 안에 |직립 오차| <= 5°, |진자 각속도| <= 0.5 rad/s 를 5 s 연속 (조원 B 와 같은 기준)
    느슨  마지막 2 s 의 90% 이상을 |직립 오차| < 10°, |진자 각속도| < 2 rad/s (이 저장소의 기존 기준)
종료(팔 > 1100 dps, 진자 > 25 rad/s)는 실패.
"""
import os
import sys
import json
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s2r_env import S2REnv, NOMINAL  # noqa: E402

CONDITIONS = {
    "nom": {},
    "J_lo": dict(J=0.0210),
    "J_hi": dict(J=0.0260),
    "fric_lo": dict(Fp=0.45, Fm=0.45, b=0.08),
    "fric_hi": dict(Fp=0.85, Fm=0.85, b=0.25),
    "k_lo": dict(k=0.80),
    "k_hi": dict(k=0.95),
    "kt_lat": dict(kt=0.90, lat=2),
}
STARTS = [(th, math.radians(a)) for th in (0.0, 1.0) for a in (-30, -15, -5, -2, 2, 5, 15, 30)]
EP_SEC = 12.0
HOLD_SEC = 5.0


def run_episode(env, policy, th0, al0):
    obs = env.set_state(th0, al0, 0.0, 0.0)
    n = env.max_steps
    phi = np.full(n, math.pi)
    ald = np.full(n, 99.0)
    taus = np.zeros(n)
    ret = 0.0
    alive = n
    for i in range(n):
        obs, r, term, _tr, info = env.step(np.array([policy(obs)], dtype=np.float32))
        ret += r
        phi[i], ald[i], taus[i] = info["phi"], info["ald"], info["tau"]
        if term:
            alive = i + 1
            break
    ok = (np.abs(phi) <= math.radians(5.0)) & (np.abs(ald) <= 0.5)
    need = int(round(HOLD_SEC / env.ctrl_dt))
    run = 0
    strict = False
    t_hold = None
    for i in range(alive):
        run = run + 1 if ok[i] else 0
        if run >= need:
            strict = True
            t_hold = (i + 1 - need) * env.ctrl_dt
            break
    tail = int(round(2.0 / env.ctrl_dt))
    up = (np.abs(phi) < math.radians(10.0)) & (np.abs(ald) < 2.0)
    loose = alive == n and float(np.mean(up[-tail:])) > 0.9
    near = np.flatnonzero(np.abs(phi) < math.radians(30.0))
    return dict(strict=bool(strict), loose=bool(loose), t_hold=t_hold, ret=ret,
                t_near=float(near[0] * env.ctrl_dt) if near.size else None,
                terminated=alive < n, sat=float(np.mean(np.abs(taus[:alive]) > 2.4)))


def evaluate(policy, env_cls=S2REnv, env_kw=None, conditions=None, verbose=True):
    """policy: obs(np.float32, 6) -> u (float, -1..1). 반환: 조건별 통과 수와 합계."""
    env_kw = dict(env_kw or {})
    conds = conditions or CONDITIONS
    out = {}
    tot_s = tot_l = 0
    for name, over in conds.items():
        env = env_cls(params=dict(NOMINAL, **over), episode_sec=EP_SEC, init="hang", **env_kw)
        env.reset(seed=0)
        rows = [run_episode(env, policy, th, al) for th, al in STARTS]
        s = sum(r["strict"] for r in rows)
        lo = sum(r["loose"] for r in rows)
        th = [r["t_hold"] for r in rows if r["t_hold"] is not None]
        out[name] = dict(strict=s, loose=lo, t_hold_med=float(np.median(th)) if th else None,
                         ret=float(np.mean([r["ret"] for r in rows])),
                         terminated=sum(r["terminated"] for r in rows))
        tot_s += s
        tot_l += lo
        if verbose:
            print("  %-8s 엄격 %2d/16  느슨 %2d/16  직립 시작 중앙 %s  종료 %d" % (
                name, s, lo, ("%.2f s" % np.median(th)) if th else "  -  ", out[name]["terminated"]))
    out["total_strict"] = tot_s
    out["total_loose"] = tot_l
    if verbose:
        print("  합계 엄격 %d/128, 느슨 %d/128" % (tot_s, tot_l))
    return out


def main():
    import validate_closedloop as V
    actor = V.ActorNP(os.path.join(HERE, "pi_mirror_20261124", "pi_host", "actor_rev11.npz"))
    pol = lambda o: actor(o.astype(np.float64))  # noqa: E731
    res = {}
    print("기준선 1: 원래 sac_rev11 정책 단독 (보상 없음, 토크 2.52 N m 상한)")
    res["orig_direct"] = evaluate(pol)

    class PerfectBridge(V.BridgeEnv):
        """보상 값 = 플랜트 마찰 그대로 (측정이 완벽한 경우, 실기보다 낙관적)."""

        def __init__(self, params=None, **kw):
            p = dict(params or {})
            super().__init__(dict(Fc_pos=p.get("Fp", NOMINAL["Fp"]), Fc_neg=p.get("Fm", NOMINAL["Fm"]),
                                  viscous=p.get("b", NOMINAL["b"])), params=p, **kw)

    print("\n기준선 2: 원래 sac_rev11 + 브리지 보상 (c) (실기에서 28/28 잡은 조합)")
    res["orig_bridge"] = evaluate(pol, env_cls=PerfectBridge)
    with open(os.path.join(HERE, "baseline_eval.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
