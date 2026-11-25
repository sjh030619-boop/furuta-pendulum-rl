"""앱 방식 그대로: 미리 학습한 정책에서 출발해 학생 보상으로 CEM '고쳐 배우기' (rev11 실기 보정 환경).

    python s2r/cem_finetune.py --base a.npz b.npz          # 바탕 정책마다 학생 보상 전부 (방식 robust, 선택 gate)
    python s2r/cem_finetune.py --reward height_effort --mode app --select best     # 앱 그대로
    python s2r/cem_finetune.py --report

CEM 은 app/src/js/learn.js 를 그대로 옮겼습니다(세대 24, 한 세대 20개, 상위 5개 평균, 변이 0.06, 세대마다 x0.93,
시작 5개: (θ, α) = (0,0) (0,±0.35) (±0.2,±0.1), 점수 = 에피소드 보상 합의 평균).
다른 점은 물리뿐입니다: 앱의 rev18b 식 대신 S2REnv(rev11 보정, 실기 호스트와 같은 관측·토크 경로), 에피소드 8 s.

방식 app    앱 그대로: 공칭 조건 하나에서 시작 5개.
방식 robust 시작 5개를 조건 5개(공칭, 마찰+, 마찰-, 토크0.9+지연, 결합0.80)와 하나씩 짝지음. 계산량은 같음.

선택 best   앱 그대로: 세대 통틀어 점수 최고였던 후보 하나.
선택 gate   후보 넷(최고 후보, 마지막 평균, 8·16세대 평균)을 고정 시험 32칸에 넣어 30 이상인 것 중 학생 점수 최고.
            하나도 없거나 통과한 후보의 점수가 바탕보다 낮으면 바탕 정책 유지. 고정 시험은 보상과 무관(조건 8개 x 시작 4개, 엄격 기준).

예행연습(runs_reward_pilot, SAC 바탕 1개, 앱 기준 20 s, 선택 best): 정상 보상 11개 중 128칸 120 이상 유지가
방식 app 2/11, robust 7/11. 그래서 기본을 robust + gate 로 했습니다.

안전 정지(팔 > 1100 dps, 진자 > 25 rad/s)로 에피소드가 끊기면, 남은 시간은 '진자가 매달려 멈춘 상태'의 점수로 채웁니다
(실기에서 안전 정지 뒤 실제로 그렇게 되므로). 이렇게 안 하면 감점만 있는 규칙에서 일찍 끊기는 쪽이 이득이 됩니다.

예산은 시뮬 스텝 수로 셉니다. 앱(JS)은 이 PC 에서 21 µs/스텝(측정)이라, 기본 설정 96만 스텝 = 앱에서 약 20 s.
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
OUT = os.path.join(HERE, "runs_reward")
BASE = os.path.join(HERE, "policies", "stage1_sac_s0.npz")
APP_US_PER_STEP = 21.2
GATE = 30

STARTS = [(0.0, 0.00), (0.0, 0.35), (0.0, -0.35), (0.2, 0.10), (-0.2, -0.10)]
ROBUST = [{}, dict(Fp=0.85, Fm=0.85, b=0.25), dict(Fp=0.45, Fm=0.45, b=0.08), dict(kt=0.90, lat=2), dict(k=0.80)]
EP_SEC = 8.0
SHAPES = [(64, 6), (64,), (64, 64), (64,), (1, 64), (1,)]
KEYS = {"sac": ["latent_pi.0.weight", "latent_pi.0.bias", "latent_pi.2.weight", "latent_pi.2.bias", "mu.weight", "mu.bias"],
        "ppo": ["pi.0.weight", "pi.0.bias", "pi.2.weight", "pi.2.bias", "action.weight", "action.bias"]}

_ENVS = {}


def kind_of(npz_path):
    w = np.load(npz_path)
    return str(w["kind"]) if "kind" in w.files else "sac"


def flatten(npz_path):
    w = np.load(npz_path)
    return np.concatenate([w[k].astype(np.float64).ravel() for k in KEYS[kind_of(npz_path)]])


def unflatten(v):
    out, k = [], 0
    for sh in SHAPES:
        n = int(np.prod(sh))
        out.append(v[k:k + n].reshape(sh))
        k += n
    return out


def actor(v, kind="sac"):
    """SAC: ReLU 두 층 + tanh 출력.  PPO: tanh 두 층 + 자르기 (policy_np.ActorNP, 앱 policy.js 와 같음)."""
    W1, b1, W2, b2, Wm, bm = unflatten(v)
    if kind == "ppo":
        def pol(o):
            h = np.tanh(W1 @ o + b1)
            h = np.tanh(W2 @ h + b2)
            return float(np.clip((Wm @ h + bm)[0], -1.0, 1.0))
    else:
        def pol(o):
            h = np.maximum(0.0, W1 @ o + b1)
            h = np.maximum(0.0, W2 @ h + b2)
            return float(np.tanh((Wm @ h + bm)[0]))
    return pol


def save_npz(v, path, kind="sac"):
    d = dict(zip(KEYS[kind], unflatten(v)))
    if kind == "ppo":
        d["kind"] = np.array("ppo")
    np.savez(path, **d)


def _env(ci, mode, sec=EP_SEC):
    from s2r_env import S2REnv, NOMINAL
    key = (ci if mode == "robust" else 0, sec)
    if key not in _ENVS:
        e = S2REnv(params=dict(NOMINAL, **(ROBUST[key[0]])), episode_sec=sec, init="hang")
        e.reset(seed=0)
        _ENVS[key] = e
    return _ENVS[key]


def score(args):
    """후보 하나의 점수(앱 learn.js score 와 같은 뜻). 워커 프로세스에서 돕니다."""
    v, rkey, mode, kind = args
    from student_rewards import BY_KEY
    fn = BY_KEY[rkey]["fn"]
    pol = actor(v, kind)
    total = 0.0
    for i, (th0, al0) in enumerate(STARTS):
        env = _env(i, mode)
        o = env.set_state(th0, al0, 0.0, 0.0)
        n = env.max_steps
        for k in range(n):
            u = pol(o.astype(np.float64))
            o, _r, term, _tr, info = env.step(np.array([u], dtype=np.float32))
            th = float(env.data.qpos[0])
            total += fn(info["phi"], info["ald"], float(env.data.qvel[0]), u, th)
            if term:
                total += (n - k - 1) * fn(math.pi, 0.0, 0.0, 0.0, th)
                break
    return total / len(STARTS)


def gate32(args):
    """고정 시험 32칸(조건 8개 x 시작 4개, 엄격 기준). 보상과 무관. 30 이상이어야 실기에 올릴 수 있음."""
    v, kind = args
    from evaluate import run_episode, CONDITIONS, STARTS as EV_STARTS
    from s2r_env import S2REnv, NOMINAL
    pol = actor(v, kind)

    def f(o):
        return pol(o.astype(np.float64))
    n = 0
    for name, over in CONDITIONS.items():
        key = ("gate", name)
        if key not in _ENVS:
            e = S2REnv(params=dict(NOMINAL, **over), episode_sec=12.0, init="hang")
            e.reset(seed=0)
            _ENVS[key] = e
        for i in (0, 5, 10, 15):
            n += run_episode(_ENVS[key], f, EV_STARTS[i][0], EV_STARTS[i][1])["strict"]
    return n


def behavior(v, kind="sac"):
    """공칭 조건, 매달린 8개 시작에서 12 s: 세우는 시각, 서 있을 때 전류·팔 움직임, 팔이 간 거리."""
    from s2r_env import S2REnv, NOMINAL
    env = S2REnv(params=dict(NOMINAL), episode_sec=12.0, init="hang")
    env.reset(seed=0)
    pol = actor(v, kind)
    t_up, iq, thd_h, drift, u2, stops = [], [], [], [], [], 0
    for a in (30, -30, 15, -15, 5, -5, 2, -2):
        o = env.set_state(0.0, math.radians(a))
        run, first = 0, None
        for k in range(env.max_steps):
            u = pol(o.astype(np.float64))
            o, _r, term, _tr, info = env.step(np.array([u], dtype=np.float32))
            u2.append(u * u)
            ok = abs(info["phi"]) <= math.radians(5.0) and abs(info["ald"]) <= 0.5
            run = run + 1 if ok else 0
            if first is None and run >= 250:
                first = (k + 1 - 250) * env.ctrl_dt
            if k >= 350:
                iq.append(abs(info["tau"]) / 2.1)
                thd_h.append(abs(float(env.data.qvel[0])))
            if term:
                stops += 1
                break
        drift.append(abs(float(env.data.qpos[0])))
        if first is not None:
            t_up.append(first)
    return dict(stood=len(t_up), t_up_med=float(np.median(t_up)) if t_up else None,
                hold_iq=float(np.mean(iq)) if iq else None, hold_arm_speed=float(np.mean(thd_h)) if thd_h else None,
                arm_drift=float(np.mean(drift)), effort=float(np.mean(u2)), safety_stops=stops)


def gate128(v, kind="sac"):
    from evaluate import evaluate
    pol = actor(v, kind)
    r = evaluate(lambda o: pol(o.astype(np.float64)), verbose=False)
    return r["total_strict"], {k: x["strict"] for k, x in r.items() if isinstance(x, dict)}


def cem(pool, base, rkey, mode, gens=24, pop=20, elite=5, sigma=0.06, decay=0.93, seed=12345, select="gate"):
    kind = kind_of(base)
    rng = np.random.default_rng(seed)
    mean = flatten(base)
    v0 = mean.copy()
    base_score = pool.map(score, [(mean, rkey, mode, kind)])[0]
    best, best_s, hist, keep = mean.copy(), -np.inf, [], {}
    t0 = time.perf_counter()
    for g in range(1, gens + 1):
        cand = [mean + sigma * rng.standard_normal(mean.size) for _ in range(pop)]
        sc = pool.map(score, [(c, rkey, mode, kind) for c in cand])
        order = np.argsort(sc)[::-1]
        mean = np.mean([cand[i] for i in order[:elite]], axis=0)
        sigma *= decay
        if sc[order[0]] > best_s:
            best_s, best = sc[order[0]], cand[order[0]].copy()
        hist.append(float(sc[order[0]]))
        if g in (gens // 3, 2 * gens // 3):
            keep["평균 %d세대" % g] = mean.copy()
    ep = int(round(EP_SEC / 0.02))
    steps = gens * pop * len(STARTS) * ep + len(STARTS) * ep
    info = dict(base_score=float(base_score), history=hist, gens=gens, pop=pop, select=select, net=kind)
    if select == "best":
        chosen, chosen_s, picked, gate = best, best_s, "최고 후보", None
    else:
        keep["최고 후보"] = best
        keep["마지막 평균"] = mean
        names = list(keep)
        sc = pool.map(score, [(keep[n], rkey, mode, kind) for n in names])
        gt = pool.map(gate32, [(keep[n], kind) for n in names])
        steps += len(names) * (len(STARTS) * ep + 32 * 600)
        info["candidates"] = {n: dict(score=float(a), gate32=int(b)) for n, a, b in zip(names, sc, gt)}
        ok = [i for i in range(len(names)) if gt[i] >= GATE]
        if ok and max(sc[j] for j in ok) <= base_score:
            chosen, chosen_s, picked, gate = v0, base_score, "바탕 유지(개선 없음)", None
        elif ok:
            i = max(ok, key=lambda j: sc[j])
            chosen, chosen_s, picked, gate = keep[names[i]], sc[i], names[i], gt[i]
        else:
            chosen, chosen_s, picked, gate = v0, base_score, "바탕 유지(탈락)", None
    info.update(best_score=float(chosen_s), raw_best_score=float(best_s), picked=picked, gate32=gate,
                sim_steps=steps, app_seconds=round(steps * APP_US_PER_STEP * 1e-6, 1),
                py_seconds=round(time.perf_counter() - t0, 1))
    return chosen, info


def one(pool, base, rkey, mode, out, **kw):
    from student_rewards import BY_KEY
    best, info = cem(pool, base, rkey, mode, **kw)
    kind = info["net"]
    bname = os.path.splitext(os.path.basename(base))[0]
    tag = "%s__%s__%s" % (bname, rkey, mode)
    save_npz(best, os.path.join(out, tag + ".npz"), kind)
    beh, (g128, per) = pool.apply(behavior, (best, kind)), pool.apply(gate128, (best, kind))
    rec = dict(base=bname, reward=rkey, name=BY_KEY[rkey]["name"], kind=BY_KEY[rkey]["kind"], mode=mode, gate128=g128,
               per=per, behavior=beh, **info)
    with open(os.path.join(out, tag + ".json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print("%-14s %-22s 점수 %8.1f -> %8.1f | %-14s | 128칸 %3d | 세움 %d/8 %s | 전류 %s | 팔 이동 %.2f rad | 앱 %.0f s" % (
        bname, rkey, info["base_score"], info["best_score"], info["picked"], g128, beh["stood"],
        ("%.2f s" % beh["t_up_med"]) if beh["t_up_med"] is not None else "  -   ",
        ("%.2f A" % beh["hold_iq"]) if beh["hold_iq"] is not None else "-", beh["arm_drift"],
        info["app_seconds"]), flush=True)
    return rec


def base_record(pool, base, out):
    v, kind = flatten(base), kind_of(base)
    bname = os.path.splitext(os.path.basename(base))[0]
    beh, (g128, per) = pool.apply(behavior, (v, kind)), pool.apply(gate128, (v, kind))
    rec = dict(base=bname, reward="(바탕 정책)", gate128=g128, per=per, behavior=beh, net=kind)
    with open(os.path.join(out, "_base__%s.json" % bname), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print("%-14s 바탕 정책: 128칸 %d | 세움 %d/8 %s | 전류 %s | 팔 이동 %.2f rad" % (
        bname, g128, beh["stood"], ("%.2f s" % beh["t_up_med"]) if beh["t_up_med"] is not None else "-",
        ("%.2f A" % beh["hold_iq"]) if beh["hold_iq"] is not None else "-", beh["arm_drift"]), flush=True)


def report(out):
    import glob
    rows = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(out, "*__*__*.json")))]
    bases = sorted({r["base"] for r in rows})
    print("%-14s %5s | 정상 보상: %9s %9s %8s | 잘못된 보상: %s" % ("바탕", "128칸", "시험 통과", "128≥120", "128 최저", "걸러짐"))
    for b in bases:
        bj = json.load(open(os.path.join(out, "_base__%s.json" % b), encoding="utf-8"))
        good = [r for r in rows if r["base"] == b and r["kind"] != "bad"]
        bad = [r for r in rows if r["base"] == b and r["kind"] == "bad"]
        passed = [r for r in good if not r["picked"].startswith("바탕")]
        print("%-14s %5d | %19s %9s %8s | %s" % (
            b, bj["gate128"], "%d/%d" % (len(passed), len(good)),
            "%d/%d" % (sum(r["gate128"] >= 120 for r in passed), len(passed)),
            min([r["gate128"] for r in passed]) if passed else "-",
            "%d/%d" % (sum(r["picked"].startswith("바탕") for r in bad), len(bad))))
    print()
    for r in rows:
        h = r["behavior"]

        def f(x, p="%.2f"):
            return (p % x) if x is not None else "-"
        print("%-14s %-22s %-5s %9.1f %9.1f %-14s %4d %6s %6s %6s %6s" % (
            r["base"], r["reward"], r["kind"], r["base_score"], r["best_score"], r["picked"], r["gate128"],
            f(h["t_up_med"]), f(h["hold_iq"]), f(h["hold_arm_speed"]), f(h["arm_drift"])))


def main():
    from multiprocessing import Pool
    from student_rewards import REWARDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", default=[BASE])
    ap.add_argument("--reward", nargs="+")
    ap.add_argument("--mode", default="robust", choices=["app", "robust"])
    ap.add_argument("--select", default="gate", choices=["gate", "best"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--gens", type=int, default=24)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--elite", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.report:
        report(a.out)
        return
    kw = dict(gens=a.gens, pop=a.pop, elite=a.elite, sigma=a.sigma, select=a.select)
    keys = a.reward or [r["key"] for r in REWARDS]
    with Pool(a.workers) as pool:
        for base in a.base:
            base_record(pool, base, a.out)
            for k in keys:
                one(pool, base, k, a.mode, a.out, **kw)


if __name__ == "__main__":
    main()
