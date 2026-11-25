"""Build the wall-clock comparison figure and the stability table from rl/runs/*.json.

    python rl/plot_results.py --runs rl/runs --out rl/figures
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DASHES = [(None, None), (5, 1.6), (1.4, 1.4), (6, 1.6, 1.4, 1.6), (3, 1.4), (None, None), (5, 1.6)]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
GRID = "#e4e3de"
SURFACE = "#ffffff"

ORDER = ["ppo", "a2c", "sac", "sac@utd1", "td3", "ddpg", "tqc", "tqc@utd1",
         "crossq", "crossq@utd1"]
PRETTY = {"ppo": "PPO", "a2c": "A2C", "sac": "SAC", "td3": "TD3", "ddpg": "DDPG",
          "tqc": "TQC", "crossq": "CrossQ",
          "sac@utd1": "SAC  UTD 1", "tqc@utd1": "TQC  UTD 1",
          "crossq@utd1": "CrossQ  UTD 1"}
FAMILY = {"ppo": "on-policy", "a2c": "on-policy", "sac": "off-policy UTD 0.125",
          "td3": "off-policy UTD 0.125", "ddpg": "off-policy UTD 0.125",
          "tqc": "off-policy UTD 0.125", "crossq": "off-policy UTD 0.125",
          "sac@utd1": "off-policy UTD 1", "tqc@utd1": "off-policy UTD 1",
          "crossq@utd1": "off-policy UTD 1"}

KO = {
    "title": "Furuta 진자 스윙업+밸런싱: 강화학습 알고리즘 벽시계 시간 비교",
    "sub": "%s · 학습 시간 예산 %ds · 시드 %d개 · 평가 시간은 시계에서 제외",
    "x": "학습 벽시계 시간 [s]",
    "y1": "평가 리턴 (고정 초기조건 5개 평균)",
    "y2": "성공률 [%]  (마지막 2초 도립 유지)",
    "y3": "",
    "solved": "성공 기준 80%",
    "t_solve": "최초 성공까지 총 소요 시간 [s]  (셋업 + 학습)",
    "never": "미달성",
    "legend": "알고리즘",
    "setup": "셋업",
    "train": "학습",
    "limit": "3분",
    "baseline": "고전 제어기 (에너지 스윙업 + LQR) — 학습 0초",
    "baseline_short": "LQR+에너지",
    "no_train": "0 s  (학습 없음, 설계 시간만)",
    "note": "성공 = 5초 에피소드의 마지막 2초를 도립 ±10°, |진자각속도| < 2 rad/s 로 유지 · 점 = 개별 시드",
}
EN = {
    "title": "Furuta pendulum swing-up + balance: wall-clock RL algorithm comparison",
    "sub": "%s | %ds training budget | %d seeds | evaluation time excluded from the clock",
    "x": "training wall-clock time [s]",
    "y1": "evaluation return (mean of 5 fixed starts)",
    "y2": "success rate [%] (upright held for final 2 s)",
    "y3": "",
    "solved": "80% success",
    "t_solve": "total time to first success [s] (setup + training)",
    "never": "not reached",
    "legend": "algorithm",
    "setup": "setup",
    "train": "training",
    "limit": "3 min",
    "baseline": "classical controller (energy swing-up + LQR) - no training",
    "baseline_short": "LQR+energy",
    "no_train": "0 s  (no training, design only)",
    "note": "success = final 2 s of a 5 s episode held within 10 deg of upright with |alpha_dot| < 2 rad/s; dots = individual seeds",
}


def pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic", "Noto Sans KR"]:
        if name in have:
            return name, KO
    return None, EN


def load(runs_dirs, only=None):
    data = {}
    for d in runs_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            r = json.load(open(f))
            key = r["algo"] + ("@" + r["tag"] if r.get("tag") else "")
            if only and key not in only:
                continue
            data.setdefault(key, []).append(r)
    return data


def curve(rec, grid, key):
    t = np.array([p["t"] for p in rec["points"]], dtype=float)
    y = np.array([p[key] for p in rec["points"]], dtype=float)
    idx = np.searchsorted(t, grid, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]


def stack(recs, grid, key):
    return np.vstack([curve(r, grid, key) for r in recs])


def metrics(recs, grid):
    succ = stack(recs, grid, "success")
    ret = stack(recs, grid, "ret")
    tail = grid >= grid[-1] * 0.8
    t_solve = []
    for row in succ:
        hit = np.nonzero(row >= 0.8)[0]
        t_solve.append(grid[hit[0]] if len(hit) else np.nan)
    t_solve = np.array(t_solve, dtype=float)
    setup = np.array([r.get("setup_time", 0.0) for r in recs], dtype=float)
    t_total = t_solve + setup
    mret = ret.mean(axis=0)
    runmax = np.maximum.accumulate(mret)
    drop = runmax - mret
    gain = runmax[-1] - mret[0]
    span = gain if gain > 50.0 else float("nan")
    return dict(
        n_seeds=len(recs),
        t_solve=t_solve,
        t_total=t_total,
        setup=float(setup.mean()),
        n_solved=int(np.sum(~np.isnan(t_solve))),
        t_solve_med=float(np.nanmedian(t_solve)) if np.any(~np.isnan(t_solve)) else float("nan"),
        t_total_med=float(np.nanmedian(t_total)) if np.any(~np.isnan(t_total)) else float("nan"),
        final_success=float(succ[:, tail].mean()),
        final_ret=float(ret[:, tail].mean()),
        seed_spread=float(np.mean(succ.max(axis=0) - succ.min(axis=0))),
        drawdown=float(drop.max()),
        drawdown_pct=float(drop.max() / span * 100.0),
        fps=int(np.mean([r["fps"] for r in recs])),
        steps=int(np.mean([r["steps"] for r in recs])),
        device=recs[0]["device"],
        n_envs=recs[0]["n_envs"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=[os.path.join(HERE, "runs")])
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--title", default="")
    p.add_argument("--out", default=os.path.join(HERE, "figures"))
    p.add_argument("--name", default="furuta_rl_wallclock")
    p.add_argument("--baseline", default=os.path.join(HERE, "baseline.json"))
    a = p.parse_args()

    base = None
    if a.baseline and os.path.exists(a.baseline):
        base = json.load(open(a.baseline))

    font, L = pick_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK3,
        "ytick.color": INK3,
        "font.size": 10,
    })

    data = load(a.runs, a.only)
    algos = [k for k in ORDER if k in data] + [k for k in sorted(data) if k not in ORDER]
    if not algos:
        print("no runs found in", a.runs)
        return
    budget = max(r["budget"] for recs in data.values() for r in recs)
    grid = np.arange(0.0, budget + 1e-9, 0.5)
    nseeds = max(len(v) for v in data.values())
    style = {k: (SLOTS[i % len(SLOTS)], DASHES[i % len(DASHES)], MARKERS[i % len(MARKERS)])
             for i, k in enumerate(algos)}
    M = {k: metrics(data[k], grid) for k in algos}

    fig = plt.figure(figsize=(13.5, 10.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.30, wspace=0.22,
                          left=0.07, right=0.985, top=0.885, bottom=0.085)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
        ax.grid(True, color=GRID, lw=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(0, budget)
        ax.set_xlabel(L["x"])

    every = max(1, len(grid) // 14)
    for k in algos:
        col, dash, mk = style[k]
        for ax, key, scale in ((ax1, "ret", 1.0), (ax2, "success", 100.0)):
            y = stack(data[k], grid, key) * scale
            m = y.mean(axis=0)
            ax.fill_between(grid, y.min(axis=0), y.max(axis=0), color=col, alpha=0.13, lw=0)
            ln, = ax.plot(grid, m, color=col, lw=2.0, solid_capstyle="round",
                          marker=mk, markevery=every, markersize=5.5,
                          markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
            if dash[0] is not None:
                ln.set_dashes(list(dash))

    if base is not None:
        bp = base["points"][0]
        for ax, val in ((ax1, bp["ret"]), (ax2, bp["success"] * 100.0)):
            ax.axhline(val, color=INK2, lw=1.6, ls=(0, (7, 3)), zorder=2)
        ax1.annotate(L["baseline"], xy=(budget * 0.015, bp["ret"]), xytext=(0, -14),
                     textcoords="offset points", ha="left", va="top",
                     color=INK2, fontsize=9, fontweight="bold")

    allmin = []
    for k in algos:
        allmin.append(np.percentile(stack(data[k], grid, "ret").min(axis=0), 3))
    floor = float(np.min(allmin))
    top = max(np.max(stack(data[k], grid, "ret")) for k in algos)
    pad = (top - floor) * 0.08
    ax1.set_ylim(floor - pad, top + pad)

    ax1.set_ylabel(L["y1"])
    ax2.set_ylabel(L["y2"])
    ax2.set_ylim(-4, 104)
    ax2.axhline(80, color=INK3, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax2.annotate(L["solved"], xy=(budget * 0.985, 80), xytext=(0, -5),
                 textcoords="offset points", ha="right", va="top",
                 color=INK3, fontsize=8.5)

    handles = [Line2D([0], [0], color=style[k][0], lw=2.4, marker=style[k][2], markersize=5.5,
                      markeredgecolor=SURFACE,
                      dashes=list(style[k][1]) if style[k][1][0] is not None else (None, None),
                      label="%-7s %s" % (PRETTY.get(k, k), FAMILY.get(k, "")))
               for k in algos]
    ax1.legend(handles=handles, loc="lower right", frameon=False, ncols=2,
               labelspacing=0.45, handlelength=2.6, fontsize=9.5,
               title=L["legend"], title_fontproperties={"size": 9, "weight": "bold"},
               alignment="left")

    ax3.set_axisbelow(True)
    ax3.grid(True, axis="x", color=GRID, lw=0.7)
    for s in ("top", "right", "left"):
        ax3.spines[s].set_visible(False)
    nrow = len(algos) + (1 if base is not None else 0)
    ys = np.arange(nrow)[::-1]
    ylab = [PRETTY.get(k, k) for k in algos]
    if base is not None:
        ylab.append(L["baseline_short"])
        bt = base.get("setup_time", 0.0)
        ax3.barh(ys[-1], max(bt, 0.6), color=INK2, height=0.54, lw=0, alpha=0.45)
        ax3.text(max(bt, 0.6) + budget * 0.02, ys[-1], L["no_train"], va="center",
                 ha="left", fontsize=9.5, color=INK2, fontweight="bold")
    ys = ys[:len(algos)]
    xmax = budget * 1.28
    for y, k in zip(ys, algos):
        col = style[k][0]
        med = M[k]["t_total_med"]
        setup = M[k]["setup"]
        if np.isnan(med):
            ax3.barh(y, budget, color=col, alpha=0.09, height=0.54, lw=0)
            ax3.text(budget * 0.015, y, L["never"], va="center", ha="left",
                     fontsize=9, color=INK3)
        else:
            ax3.barh(y, setup, color=col, height=0.54, lw=0, alpha=0.32)
            ax3.barh(y, med - setup, left=setup + xmax * 0.004, color=col,
                     height=0.54, lw=0, alpha=0.95)
            pts = M[k]["t_total"]
            ok = ~np.isnan(pts)
            if np.any(ok):
                ax3.scatter(pts[ok], np.full(int(ok.sum()), y), s=24, color=SURFACE,
                            edgecolors=col, linewidths=1.5, zorder=4)
            nb = int((~ok).sum())
            lab = "%.0f s" % med + (("  (%d/%d)" % (M[k]["n_solved"], M[k]["n_seeds"])) if nb else "")
            right = max(med, float(np.nanmax(pts[ok])) if np.any(ok) else med)
            ax3.text(right + xmax * 0.022, y, lab, va="center", ha="left",
                     fontsize=9.5, color=INK, fontweight="bold")
    ax3.set_yticks(np.arange(nrow)[::-1], ylab, color=INK)
    ax3.tick_params(axis="y", length=0)
    ax3.axvline(180.0, color=INK3, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax3.annotate(L["limit"], xy=(180.0, nrow - 0.55), xytext=(-4, 0),
                 textcoords="offset points", ha="right", va="center",
                 color=INK3, fontsize=8.5)
    ax3.set_xlim(0, xmax)
    ax3.set_xlabel(L["t_solve"])
    leg3 = [Line2D([0], [0], color=INK3, lw=7, alpha=0.32, label=L["setup"]),
            Line2D([0], [0], color=INK3, lw=7, alpha=0.95, label=L["train"])]
    ax3.legend(handles=leg3, loc="lower right", frameon=False, fontsize=8.5,
               handlelength=1.4, borderaxespad=0.2)

    ref = data[algos[0]][0]
    env = "armature %.0e kg·m²  ·  %.0f Hz  ·  sim dt %.0f ms  ·  %.0f s episode" % (
        ref["armature"], ref["ctrl_hz"], ref.get("timestep", 0.002) * 1000.0,
        ref["episode_steps"] / ref["ctrl_hz"])
    fig.suptitle(a.title or L["title"], x=0.07, ha="left", fontsize=15.5, fontweight="bold", y=0.975)
    fig.text(0.07, 0.932, L["sub"] % (env, int(budget), nseeds), ha="left",
             fontsize=10, color=INK2)
    fig.text(0.07, 0.022, L["note"], ha="left", fontsize=8.5, color=INK3)

    os.makedirs(a.out, exist_ok=True)
    png = os.path.join(a.out, a.name + ".png")
    fig.savefig(png, dpi=150)
    fig.savefig(os.path.join(a.out, a.name + ".svg"))
    print("wrote", png, "(font: %s)" % (font or "default/EN"))

    rows = []
    hdr = ["algo", "family", "setup[s]", "t_train[s]", "t_total[s]", "solved",
           "final_succ", "final_ret", "seed_spread", "drawdown", "fps", "steps",
           "device", "n_envs"]
    for k in algos:
        m = M[k]
        rows.append([PRETTY.get(k, k), FAMILY.get(k, ""), "%.0f" % m["setup"],
                     "-" if np.isnan(m["t_solve_med"]) else "%.0f" % m["t_solve_med"],
                     "-" if np.isnan(m["t_total_med"]) else "%.0f" % m["t_total_med"],
                     "%d/%d" % (m["n_solved"], m["n_seeds"]),
                     "%.2f" % m["final_success"], "%.0f" % m["final_ret"],
                     "%.3f" % m["seed_spread"],
                     "%.0f" % m["drawdown"] + ("" if np.isnan(m["drawdown_pct"])
                                               else " (%.0f%%)" % m["drawdown_pct"]),
                     "%d" % m["fps"], "%d" % m["steps"], m["device"], "%d" % m["n_envs"]])
    w = [max(len(str(r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(hdr))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))

    out = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
               for kk, vv in M[k].items()} for k in algos}
    with open(os.path.join(a.out, a.name + "_summary.json"), "w") as f:
        json.dump(dict(budget=budget, seeds=nseeds, metrics=out), f, indent=1)


if __name__ == "__main__":
    main()
