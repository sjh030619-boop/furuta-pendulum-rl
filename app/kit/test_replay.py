"""실기 기록 다시 넣기: kit_server 의 토크 경로와 회전 감지가 실기에서 돌았던 run_policy 와 같은지 봅니다.

  python test_replay.py <host-dir> <기록 폴더>[=<정책 npz 폴더>] ...

기록 파일 이름이 r0__<이름>__시각.csv 이고 정책 폴더에 <이름>.npz 가 있으면 정책 출력도 비교합니다.
마찰 보상(--comp)으로 돌린 기록(옆에 _fric.json 이 있음)은 토크 경로가 달라 건너뜁니다.

1. 기록의 (u, theta_dot, alpha) 를 torque_path 에 넣어 iq, tau_cmd, brake_limited 가 기록과 한 줄도 다르지 않은지.
2. 정책 npz 가 있으면 기록의 관측으로 Actor 를 돌려 u 가 기록과 같은지 (앱이 보내는 float32 형식 그대로).
3. SpinWatch 가 '계속 돈 기록'만 잡고 정상 기록은 잡지 않는지.
모터도 센서도 쓰지 않습니다.
"""
import os
import sys
import csv
import glob
import math

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kit_server as ks  # noqa: E402


def pack(npz):
    w = np.load(npz)
    parts = [w["latent_pi.0.weight"], w["latent_pi.0.bias"], w["latent_pi.2.weight"], w["latent_pi.2.bias"],
             w["mu.weight"], w["mu.bias"]]
    return np.concatenate([np.asarray(x, dtype="<f4").ravel() for x in parts]).astype(np.float64)


def main():
    args = sys.argv[1:]
    rp, session, canguard = ks.load_host(args[0])
    cfg = rp.CONFIG
    cfg["IQ_MAX"] = ks.IQ_MAX
    files = []
    for d in args[1:]:
        d, _, pd = d.partition("=")
        files += [(p, pd or None) for p in sorted(glob.glob(os.path.join(d, "*.csv")))]
    n_rows = n_iq = n_tau = n_br = n_files = n_comp = 0
    u_max = 0.0
    n_u = 0
    spins = []
    for p, pol_dir in files:
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        if not rows or "alpha" not in rows[0] or "brake_limited" not in rows[0]:
            continue
        if os.path.exists(p[:-4] + "_fric.json"):
            n_comp += 1
            continue
        n_files += 1
        name = os.path.basename(p)
        actor = None
        if pol_dir:
            parts = name.split("__")
            cand = os.path.join(pol_dir, parts[1] + ".npz") if len(parts) >= 2 else None
            if cand and os.path.exists(cand):
                actor = ks.Actor(np, [6, 64, 64, 1], "relu", True, pack(cand))
        spin = ks.SpinWatch()
        fired = None
        for r in rows:
            u, thd, al = float(r["u"]), float(r["theta_dot"]), float(r["alpha"])
            iq, tau, _, br = ks.torque_path(rp, cfg, u, thd, al)
            n_rows += 1
            n_iq += iq != int(r["iq"])
            n_tau += abs(tau - float(r["tau_cmd"])) > 6e-5
            n_br += br != int(r["brake_limited"])
            if actor is not None:
                th, ald = float(r["theta"]), float(r["alpha_dot"])
                o = [math.cos(th), math.sin(th), math.cos(al), math.sin(al), thd / 20.0, ald / 40.0]
                u_max = max(u_max, abs(actor(o) - u))
                n_u += 1
            if fired is None and spin.update(float(r["t"]), al):
                fired = float(r["t"])
        un = [float(rows[0]["alpha"])]
        for a, b in zip(rows[1:], rows[:-1]):
            un.append(un[-1] + rp.wrap(float(a["alpha"]) - float(b["alpha"])))
        turns = abs(un[-1] - un[0]) / (2 * math.pi)
        spins.append((name, fired, turns))
    print("기록 %d개, %d줄 (마찰 보상 기록 %d개는 건너뜀)" % (n_files, n_rows, n_comp))
    print("  iq 다른 줄 %d, tau_cmd 다른 줄 %d, 제동 제한 표시 다른 줄 %d" % (n_iq, n_tau, n_br))
    if n_u:
        print("  정책 출력 u: %d줄 비교, 최대 차이 %.2e" % (n_u, u_max))
    hit = [s for s in spins if s[1] is not None]
    print("  회전 감지: %d개 기록에서 걸림" % len(hit))
    for name, t, turns in hit:
        print("    %-36s %.2f s 에 정지 (그 기록의 전체 순회전 %.1f 바퀴)" % (name, t, turns))
    miss = [s for s in spins if s[1] is None and s[2] > 3]
    wrong = [s for s in hit if s[2] < 3]
    print("  놓친 기록(3바퀴 넘게 돌았는데 안 걸림) %d, 잘못 걸린 기록 %d" % (len(miss), len(wrong)))
    ok = n_iq == 0 and n_tau == 0 and n_br == 0 and not miss and not wrong and u_max < 1e-5
    print("통과" if ok else "실패")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
