"""동역학 기반 제어기를 PPO에 붙이는 방식들을 한 장으로.

    python rl/plot_prior.py

왼쪽은 시간에 따른 평가 리턴, 오른쪽은 처음으로 성공에 닿기까지 걸린 총 시간입니다.
점선은 제어기 자체의 성능 — 학습 0초로 이미 거기에 있으므로, 그 위로 올라가야 RL을 쓴 값을 합니다.
"""
import os
import io
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

INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
GRID = "#e4e3de"
SURFACE = "#ffffff"

STYLE = [
    ("scratch", "#8a8880", (None, None), "PPO 그대로", "plain PPO"),
    ("residual", "#2a78d6", (None, None), "잔차 학습", "residual"),
    ("corerl9", "#eb6834", (5, 1.6), "제어 정규화 λ=9", "CORE-RL λ=9"),
    ("corerl", "#eda100", (2, 1.6), "제어 정규화 λ=3", "CORE-RL λ=3"),
    ("jsrl", "#1baf7a", (6, 1.6, 1.4, 1.6), "Jump-Start", "Jump-Start"),
    ("bc", "#e87ba4", (1.4, 1.4), "BC 사전학습", "BC pretrain"),
    ("dapg", "#6250d6", (4, 1.4), "DAPG", "DAPG"),
]

KO = dict(
    title="동역학 기반 제어기를 PPO에 붙이는 네 가지 방식",
    sub="출발 제어기 = 에너지 스윙업 + LQR (모델에서 계산, 설계 0.06초) · 보상·환경·PPO 설정은 벤치마크와 동일 · 시드 3개 · 띠 = 시드 최소~최대",
    p1="시간에 따른 평가 리턴", p2="처음 성공까지 걸린 총 시간 [s]  (사전학습 시간 포함)",
    x="학습 벽시계 시간 [s]", y="평가 리턴 (고정 초기조건 5개 평균)",
    base="제어기 자체 — 학습 0초", never="미달성",
    note="성공 = 5초 에피소드의 마지막 2초를 도립 ±10°, |진자각속도| < 2 rad/s 로 유지 · 점 = 개별 시드",
)
EN = dict(
    title="Four ways to attach a model-based controller to PPO",
    sub="prior = energy swing-up + LQR computed from the model | same reward, env and PPO settings as the benchmark | 3 seeds",
    p1="evaluation return over time", p2="total time to first success [s] (incl. pretraining)",
    x="training wall-clock time [s]", y="evaluation return (mean of 5 fixed starts)",
    base="the controller alone - no training", never="not reached",
    note="success = final 2 s of a 5 s episode held within 10 deg of upright",
)


def pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic", "Noto Sans KR"]:
        if n in have:
            return n, KO
    return None, EN


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        out.setdefault(r["mode"], []).append(r)
    return out


def curves(recs, grid, key="ret"):
    cs = []
    for r in recs:
        t = np.array([p["t"] for p in r["points"]])
        v = np.array([p[key] for p in r["points"]], dtype=float)
        idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
        c = v[idx]
        c[grid < t[0]] = np.nan
        cs.append(c)
    return np.array(cs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(HERE, "runs_prior"))
    p.add_argument("--baseline", default=os.path.join(HERE, "baseline.json"))
    p.add_argument("--out", default=os.path.join(HERE, "figures"))
    p.add_argument("--name", default="furuta_rl_prior")
    a = p.parse_args()

    font, L = pick_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK3, "ytick.color": INK3, "font.size": 10,
    })

    data = load(a.runs)
    if not data:
        print("no runs in", a.runs)
        return
    keys = [k for k in STYLE if k[0] in data]
    budget = max(r["budget"] for v in data.values() for r in v)
    grid = np.arange(0.0, budget + 1e-9, 0.25)

    base_ret = None
    if os.path.exists(a.baseline):
        b = json.load(io.open(a.baseline, encoding="utf-8"))
        base_ret = b["points"][0]["ret"]

    fig = plt.figure(figsize=(14.6, 6.9), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], left=0.062, right=0.975,
                          top=0.755, bottom=0.135, wspace=0.26)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for key, col, dash, ko, en in keys:
        c = curves(data[key], grid)
        ax1.fill_between(grid, np.nanmin(c, axis=0), np.nanmax(c, axis=0),
                         color=col, alpha=0.12, lw=0, zorder=2)
        ln, = ax1.plot(grid, np.nanmean(c, axis=0), color=col, lw=2.3, zorder=3,
                       label=(ko if font else en), solid_capstyle="round")
        if dash[0]:
            ln.set_dashes(dash)

    if base_ret is not None:
        ax1.axhline(base_ret, color=INK2, lw=1.4, ls=(0, (6, 4)), zorder=4)
        ax1.annotate(L["base"], (budget * 0.985, base_ret), xytext=(0, 7),
                     textcoords="offset points", ha="right", va="bottom",
                     fontsize=9.5, color=INK2, fontweight="bold")

    ax1.set_xlabel(L["x"], fontsize=10.5)
    ax1.set_ylabel(L["y"], fontsize=10.5)
    ax1.set_xlim(0, budget)
    ax1.grid(color=GRID, lw=0.8)
    ax1.set_axisbelow(True)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)
    allv = np.concatenate([curves(data[k[0]], grid).ravel() for k in keys])
    allv = allv[~np.isnan(allv)]
    ax1.set_ylim(np.percentile(allv, 2.0), max(allv) * 1.08)
    ax1.set_title(L["p1"], fontsize=11.5, color=INK, loc="left", pad=10, fontweight="bold")
    ax1.legend(loc="lower right", frameon=False, fontsize=9.5, labelspacing=0.45,
               handlelength=2.6, handletextpad=0.7, ncol=2, columnspacing=1.4)

    rows = []
    for key, col, _d, ko, en in keys:
        ts = [r["t_solve"] for r in data[key]]
        ok = [t for t in ts if t is not None]
        rows.append((key, col, ko if font else en, ok, len(ts),
                     float(np.mean([r["prep_s"] for r in data[key]]))))
    rows.sort(key=lambda r: (np.mean(r[3]) if len(r[3]) == r[4] else 1e9,
                             -len(r[3])))

    ys = np.arange(len(rows))[::-1]
    cap = budget * 1.02
    for y, (key, col, lab, ok, n, prep) in zip(ys, rows):
        if ok:
            m = float(np.mean(ok))
            if prep > 0.4:
                ax2.barh(y, prep, height=0.56, color=col, alpha=0.32, zorder=3)
                ax2.barh(y, max(0.0, m - prep), left=prep, height=0.56,
                         color=col, alpha=0.9, zorder=3)
            else:
                ax2.barh(y, m, height=0.56, color=col, alpha=0.9, zorder=3)
            for t in ok:
                ax2.plot([t], [y], marker="o", ms=5.5, mfc=SURFACE, mec=col, mew=1.5, zorder=5)
            txt = "%.0f s" % m if n == len(ok) else "%.0f s  (%d/%d)" % (m, len(ok), n)
            ax2.annotate(txt, (max(m, max(ok)), y), xytext=(10, 0), textcoords="offset points",
                         va="center", fontsize=10.5, fontweight="bold", color=INK)
        else:
            ax2.annotate(L["never"] + "  (0/%d)" % n, (1, y), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=10.5,
                         fontweight="bold", color="#b3302f")
    ax2.set_yticks(ys)
    ax2.set_yticklabels([r[2] for r in rows], fontsize=11)
    ax2.set_xlabel(L["p2"], fontsize=10.5)
    ax2.set_xlim(0, cap * 1.30)
    ax2.set_ylim(-0.7, len(rows) - 0.3)
    ax2.grid(axis="x", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.legend(handles=[Line2D([], [], lw=7, color=INK3, alpha=0.45,
                               label="사전학습" if font else "pretraining"),
                        Line2D([], [], lw=7, color=INK3, alpha=0.9,
                               label="합계" if font else "total")],
               loc="lower right", frameon=False, fontsize=9)
    ax2.set_title(L["p2"], fontsize=11.5, color=INK, loc="left", pad=10, fontweight="bold")
    ax2.set_xlabel("")

    fig.suptitle(L["title"], x=0.062, ha="left", fontsize=15.5, fontweight="bold", y=0.955)
    fig.text(0.062, 0.878, L["sub"], ha="left", fontsize=9.5, color=INK2)
    fig.text(0.062, 0.028, L["note"], ha="left", fontsize=8.5, color=INK3)

    os.makedirs(a.out, exist_ok=True)
    png = os.path.join(a.out, a.name + ".png")
    fig.savefig(png, dpi=150)
    fig.savefig(os.path.join(a.out, a.name + ".svg"))
    plt.close(fig)
    print("wrote", png, "(font: %s)" % (font or "default/EN"))

    print("%-9s %7s %8s %10s %8s %8s %8s" %
          ("mode", "t0_succ", "t0_ret", "solve", "prep[s]", "fin_sc", "fin_ret"))
    for key, _c, _d, _ko, _en in keys:
        v = data[key]
        ok = [r["t_solve"] for r in v if r["t_solve"] is not None]
        print("%-9s %7.2f %8.0f %10s %8.1f %8.2f %8.0f" % (
            key, np.mean([r["succ0"] for r in v]), np.mean([r["ret0"] for r in v]),
            ("%.0f" % np.mean(ok)) if len(ok) == len(v) else ("%d/%d" % (len(ok), len(v))),
            np.mean([r["prep_s"] for r in v]),
            np.mean([r["final_succ"] for r in v]), np.mean([r["final_ret"] for r in v])))


if __name__ == "__main__":
    main()
