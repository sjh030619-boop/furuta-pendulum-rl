"""2단계 실기 본선 기록 판정 (s2r/evaluate.py 와 같은 기준, 실기 기록에 적용).

    python3 analyze_stage2.py <기록 폴더>

엄격: 12 s 안에 |직립 오차| <= 5°, |진자 각속도| <= 0.5 rad/s 를 5 s 연속.
느슨: 중단 없이 끝까지 돌고 마지막 2 s 의 90% 넘게 |직립 오차| < 10°, |진자 각속도| < 2 rad/s.
잡은 시각: 처음으로 |직립 오차| < 30° 인 시각.
"""
import os
import sys
import csv
import glob
import json
import math

HOLD = 5.0
LIMIT = 12.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def judge(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    t = [float(r["t"]) for r in rows]
    t = [x - t[0] for x in t]
    phi = [wrap(float(r["alpha"]) - math.pi) for r in rows]
    ald = [float(r["alpha_dot"]) for r in rows]
    temps = [int(r["temp_C"]) for r in rows if r.get("temp_C") not in ("", None)]
    dt = (t[-1] - t[0]) / max(len(t) - 1, 1)
    need = int(round(HOLD / 0.02))
    run, t_hold = 0, None
    for i in range(len(t)):
        if t[i] > LIMIT:
            break
        ok = abs(phi[i]) <= math.radians(5.0) and abs(ald[i]) <= 0.5
        run = run + 1 if ok else 0
        if run >= need:
            t_hold = t[i + 1 - need]
            break
    tail = [i for i in range(len(t)) if t[i] >= t[-1] - 2.0]
    up = [abs(phi[i]) < math.radians(10.0) and abs(ald[i]) < 2.0 for i in tail]
    full = t[-1] >= LIMIT - 0.1
    near = next((t[i] for i in range(len(t)) if abs(phi[i]) < math.radians(30.0)), None)
    return dict(file=os.path.basename(path), strict=t_hold is not None, t_hold=t_hold,
                loose=bool(full and sum(up) / max(len(up), 1) > 0.9), t_near=near, dur=round(t[-1], 2),
                period_ms=round(1000 * dt, 2), temp_max=max(temps) if temps else None,
                iq_max=max(abs(int(r["iq"])) for r in rows if r["iq"] != ""))


def main():
    d = sys.argv[1]
    res = []
    for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
        if p.endswith(("_settle.csv", "_fric.csv")):
            continue
        r = judge(p)
        if r:
            r["arm"] = os.path.basename(p).split("__")[1] if "__" in p else "?"
            res.append(r)
            print("%-44s 엄격 %-5s 유지시작 %-6s 느슨 %-5s 잡음(30°) %-6s 길이 %5.2f s 온도 %s°C" % (
                r["file"], r["strict"], r["t_hold"] and round(r["t_hold"], 2), r["loose"],
                r["t_near"] and round(r["t_near"], 2), r["dur"], r["temp_max"]))
    arms = sorted({r["arm"] for r in res})
    print("\n%-10s %6s %6s %12s %12s" % ("팔", "엄격", "느슨", "유지시작 중앙", "잡음 중앙"))
    summ = {}
    for a in arms:
        rs = [r for r in res if r["arm"] == a]
        th = sorted(r["t_hold"] for r in rs if r["t_hold"] is not None)
        tn = sorted(r["t_near"] for r in rs if r["t_near"] is not None)
        med = lambda v: v[len(v) // 2] if v else None
        summ[a] = dict(n=len(rs), strict=sum(r["strict"] for r in rs), loose=sum(r["loose"] for r in rs),
                       t_hold_med=med(th), t_near_med=med(tn))
        print("%-10s %3d/%-2d %3d/%-2d %12s %12s" % (a, summ[a]["strict"], len(rs), summ[a]["loose"], len(rs),
                                                    summ[a]["t_hold_med"], summ[a]["t_near_med"]))
    json.dump(dict(runs=res, summary=summ), open(os.path.join(d, "stage2_summary.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
