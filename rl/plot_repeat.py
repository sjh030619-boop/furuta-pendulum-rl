"""두 번의 독립 반복(A/B)이 얼마나 일치하는지 그립니다.

    python rl/plot_repeat.py --a rl/runs --b rl/runs_b --out rl/figures

왼쪽: 최초 성공까지 걸린 시간이 반복 사이에 얼마나 움직였나 (움직인 이유는 배경 부하).
오른쪽: 안정성 지표(시드 편차)는 반복 사이에 거의 움직이지 않았나.
대각선 위에 놓이면 "두 번 다 같은 이야기", 멀면 "한 번의 측정으로 말하면 안 되는 값".
"""
import os
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

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
ORDER = ["ppo", "a2c", "sac", "td3", "ddpg", "tqc", "crossq"]
PRETTY = {"ppo": "PPO", "a2c": "A2C", "sac": "SAC", "td3": "TD3", "ddpg": "DDPG",
          "tqc": "TQC", "crossq": "CrossQ"}
COLOR = {k: SLOTS[i] for i, k in enumerate(ORDER)}

INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
GRID = "#e4e3de"
SURFACE = "#ffffff"
BUSY = "#e34948"

KO = dict(
    title="같은 벤치마크를 두 번 돌리면 무엇이 그대로이고 무엇이 바뀌는가",
    sub="1차(A)는 배경 프로그램이 간헐적으로 끼어듦 · 2차(B)는 40개 런 전부 조용함 · 각 4시드 × 180초",
    p1="최초 성공까지 걸린 시간 [s]",
    p2="시드 편차  (작을수록 누가 돌려도 같은 결과)",
    ax2x="1차 (A)", ax2y="2차 (B)",
    same="두 번이 같으면 이 선 위",
    busy="오염된 런 포함 (calib > 100 ms)",
    clean="조용한 런만",
    note="최초 성공 = 평가 성공률이 처음 80%에 닿은 시각(셋업 포함) · 시드 편차 = 시간에 걸친 시드 간 성공률 최대-최소의 평균",
    faster="빨라짐", slower="느려짐",
    zoom="확대: 두 번 다 안정한 네 개",
)
EN = dict(
    title="What repeats and what does not, across two runs of the same benchmark",
    sub="repeat A had intermittent background load; repeat B was quiet for all 40 runs; 4 seeds x 180 s each",
    p1="time to first success [s]",
    p2="seed spread (lower = same result for everyone)",
    ax2x="repeat A", ax2y="repeat B",
    same="equal on this line",
    busy="includes a contaminated run (calib > 100 ms)",
    clean="quiet runs only",
    note="first success = first time evaluation success rate reached 80% (setup included); seed spread = mean over time of max-min success across seeds",
    faster="faster", slower="slower",
    zoom="zoom: the four that are stable in both",
)


def pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic", "Noto Sans KR"]:
        if name in have:
            return name, KO
    return None, EN


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        if r.get("tag"):
            continue
        out.setdefault(r["algo"], []).append(r)
    return out


def solve_t(r, thr=0.8):
    for p in r["points"]:
        if p["success"] >= thr:
            return p["t"] + r.get("setup_time", 0.0)
    return None


def spread(recs):
    budget = max(r["budget"] for r in recs)
    grid = np.arange(0.0, budget + 1e-9, 0.5)
    curves = []
    for r in recs:
        t = np.array([p["t"] for p in r["points"]])
        s = np.array([p["success"] for p in r["points"]])
        idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
        curves.append(s[idx])
    c = np.array(curves)
    return float(np.mean(c.max(axis=0) - c.min(axis=0)))


def worst_calib(recs):
    return max(max(r.get("calib_ms_before", 0.0), r.get("calib_ms_after", 0.0)) for r in recs)


def digest(d):
    out = {}
    for a, recs in d.items():
        ts = [solve_t(r) for r in recs]
        ok = [t for t in ts if t is not None]
        out[a] = dict(t=float(np.mean(ok)) if ok else None,
                      n_solved=len(ok), n=len(recs),
                      spread=spread(recs), calib=worst_calib(recs),
                      fps=float(np.mean([r["fps"] for r in recs])))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default=os.path.join(HERE, "runs"))
    p.add_argument("--b", default=os.path.join(HERE, "runs_b"))
    p.add_argument("--out", default=os.path.join(HERE, "figures"))
    p.add_argument("--name", default="furuta_rl_repeat")
    a = p.parse_args()

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

    A = digest(load(a.a))
    B = digest(load(a.b))
    algos = [k for k in ORDER if k in A and k in B]
    algos.sort(key=lambda k: B[k]["t"] if B[k]["t"] else 1e9)

    fig = plt.figure(figsize=(14.0, 6.6), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], left=0.065, right=0.975,
                          top=0.80, bottom=0.115, wspace=0.26)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ── panel 1: paired slope, A -> B ──
    ys = np.arange(len(algos))[::-1]
    for y, k in zip(ys, algos):
        c = COLOR[k]
        ta, tb = A[k]["t"], B[k]["t"]
        dirty = A[k]["calib"] > 100.0
        ax1.plot([ta, tb], [y, y], color=c, lw=2.0, alpha=0.5, zorder=2,
                 solid_capstyle="round")
        ax1.plot([ta], [y], marker="o", ms=9, mfc=SURFACE, mec=c, mew=2.0, zorder=3)
        ax1.plot([tb], [y], marker="o", ms=9, mfc=c, mec=SURFACE, mew=1.6, zorder=4)
        if dirty:
            ax1.plot([ta], [y], marker="o", ms=15, mfc="none", mec=BUSY, mew=1.6,
                     ls="none", zorder=3)
        d = tb - ta
        lab = "%+.0f s" % d
        xr = max(ta, tb)
        ax1.annotate(lab, (xr, y), xytext=(13, 0), textcoords="offset points",
                     va="center", ha="left", fontsize=9.5, fontweight="bold",
                     color=INK2 if abs(d) < 10 else (c if d < 0 else BUSY))
    ax1.set_yticks(ys)
    ax1.set_yticklabels([PRETTY[k] for k in algos], fontsize=11, fontweight="bold", color=INK)
    ax1.set_xlabel(L["p1"], fontsize=10.5)
    ax1.set_xlim(0, max(max(A[k]["t"], B[k]["t"]) for k in algos) * 1.22)
    ax1.set_ylim(-0.7, len(algos) - 0.3)
    ax1.grid(axis="x", color=GRID, lw=0.8)
    ax1.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax1.spines[sp].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    leg1 = [Line2D([], [], marker="o", ls="none", ms=9, mfc=SURFACE, mec=INK2, mew=2.0, label="1차 (A)" if font else "repeat A"),
            Line2D([], [], marker="o", ls="none", ms=9, mfc=INK2, mec=SURFACE, mew=1.6, label="2차 (B)" if font else "repeat B"),
            Line2D([], [], marker="o", ls="none", ms=13, mfc="none", mec=BUSY, mew=1.6, label=L["busy"])]
    ax1.legend(handles=leg1, loc="upper right", frameon=False, fontsize=9,
               handletextpad=0.4, labelspacing=0.5, borderaxespad=0.1)

    # ── panel 2: A vs B scatter against the diagonal ──
    OFF = {"sac": (11, -5), "ppo": (11, 4), "tqc": (-11, 5), "crossq": (12, -4),
           "a2c": (12, -3), "td3": (-12, 7), "ddpg": (-12, 7)}
    band = 0.25

    def scatter(ax, keys, hi, labels=True, ms=11, fs=10.5):
        ax.plot([0, hi], [0, hi], color=INK3, lw=1.2, ls=(0, (4, 3)), zorder=1)
        for k in keys:
            x, y = A[k]["spread"], B[k]["spread"]
            ax.plot([x], [y], marker="o", ms=ms, mfc=COLOR[k], mec=SURFACE,
                    mew=1.8, zorder=4)
            if not labels:
                continue
            o = OFF.get(k, (11, -4))
            ax.annotate(PRETTY[k], (x, y), xytext=o, textcoords="offset points",
                        fontsize=fs, fontweight="bold", color=INK,
                        ha="left" if o[0] > 0 else "right", va="center", zorder=5)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    hi = max(max(A[k]["spread"], B[k]["spread"]) for k in algos) * 1.18 + 0.02
    far = [k for k in algos if max(A[k]["spread"], B[k]["spread"]) >= band]
    near = [k for k in algos if k not in far]
    scatter(ax2, far, hi)
    for k in near:
        ax2.plot([A[k]["spread"]], [B[k]["spread"]], marker="o", ms=7,
                 mfc=COLOR[k], mec=SURFACE, mew=1.4, zorder=4, alpha=0.55)
    ax2.annotate(L["same"], (hi * 0.33, hi * 0.33), xytext=(-7, 9),
                 textcoords="offset points", fontsize=9, color=INK3, ha="right")
    ax2.set_xlabel(L["ax2x"], fontsize=10.5)
    ax2.set_ylabel(L["ax2y"], fontsize=10.5)
    ax2.set_title(L["p2"], fontsize=10.5, color=INK2, loc="left", pad=8)

    ax2.add_patch(plt.Rectangle((0, 0), band, band, facecolor=SURFACE,
                                edgecolor=INK3, lw=1.0, ls=(0, (3, 2)), zorder=5))
    ins = ax2.inset_axes([0.46, 0.07, 0.50, 0.50])
    scatter(ins, near, band, fs=10.0, ms=10)
    ins.set_facecolor(SURFACE)
    ins.tick_params(labelsize=8.5)
    for sp in ("top", "right"):
        ins.spines[sp].set_visible(True)
        ins.spines[sp].set_color(GRID)
    ins.text(0.035, 0.955, L["zoom"], transform=ins.transAxes, fontsize=9,
             color="#0d7a56", va="top", ha="left", zorder=6)
    ax2.indicate_inset_zoom(ins, edgecolor=INK3, alpha=0.6, lw=1.0)

    fig.suptitle(L["title"], x=0.065, ha="left", fontsize=15.5, fontweight="bold", y=0.965)
    fig.text(0.065, 0.895, L["sub"], ha="left", fontsize=10, color=INK2)
    fig.text(0.065, 0.022, L["note"], ha="left", fontsize=8.5, color=INK3)

    os.makedirs(a.out, exist_ok=True)
    png = os.path.join(a.out, a.name + ".png")
    fig.savefig(png, dpi=150)
    fig.savefig(os.path.join(a.out, a.name + ".svg"))
    plt.close(fig)
    print("wrote", png, "(font: %s)" % (font or "default/EN"))

    print("%-8s %9s %9s %8s | %8s %8s | %9s %9s" %
          ("algo", "A t[s]", "B t[s]", "delta", "A sprd", "B sprd", "A calib", "B calib"))
    for k in algos:
        print("%-8s %9.0f %9.0f %+8.0f | %8.3f %8.3f | %9.0f %9.0f" %
              (PRETTY[k], A[k]["t"], B[k]["t"], B[k]["t"] - A[k]["t"],
               A[k]["spread"], B[k]["spread"], A[k]["calib"], B[k]["calib"]))


if __name__ == "__main__":
    main()
