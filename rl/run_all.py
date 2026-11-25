"""Run the whole benchmark: every algorithm x every seed, one subprocess at a time so the
wall-clock numbers are not contaminated by sibling jobs.

    python rl/run_all.py --budget 180 --seeds 0 1 2
"""
import os
import sys
import time
import json
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

ALGOS = ["ppo", "a2c", "sac", "td3", "ddpg", "tqc", "crossq"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algos", nargs="*", default=ALGOS)
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--budget", type=float, default=180.0)
    p.add_argument("--eval-every", type=float, default=6.0)
    p.add_argument("--out", default=os.path.join(HERE, "runs"))
    p.add_argument("--armature", type=float, default=5.0e-3)
    p.add_argument("--tag", default="")
    p.add_argument("--kw", default="")
    p.add_argument("--save-dir", default="")
    p.add_argument("--skip-existing", action="store_true")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    jobs = [(al, s) for al in a.algos for s in a.seeds]
    t0 = time.perf_counter()
    for i, (al, s) in enumerate(jobs, 1):
        name = "%s_s%d%s.json" % (al, s, ("_" + a.tag) if a.tag else "")
        dest = os.path.join(a.out, name)
        if a.skip_existing and os.path.exists(dest):
            print("[%2d/%2d] skip %s" % (i, len(jobs), name))
            continue
        cmd = [PY, os.path.join(HERE, "bench.py"), "--algo", al, "--seed", str(s),
               "--budget", str(a.budget), "--eval-every", str(a.eval_every),
               "--armature", str(a.armature), "--out", a.out, "--threads", "1"]
        if a.tag:
            cmd += ["--tag", a.tag]
        if a.kw:
            cmd += ["--kw", a.kw]
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            cmd += ["--save", os.path.join(a.save_dir, "%s_s%d" % (al, s))]
        print("[%2d/%2d] %-7s seed=%d ..." % (i, len(jobs), al, s), end=" ", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("FAILED")
            print(r.stdout[-2000:])
            print(r.stderr[-3000:])
        else:
            print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "ok")
    print("total %.1f min" % ((time.perf_counter() - t0) / 60.0))


if __name__ == "__main__":
    main()
