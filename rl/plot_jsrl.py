"""Jump-Start 커리큘럼을 다시 재기: 시간 기반 → 성능 기반(성공률 / 리턴 게이트).

    python rl/plot_jsrl.py

위: 제어기가 끌고 가는 구간 h 가 실제로 어떻게 내려갔는지.
아래: 그 결과로 **정책 혼자** 무엇을 할 수 있게 되었는지. jsrl 에서 제어기는 학습 보조일
뿐 배포에는 없으므로, 아래 곡선이 이 방법의 성적입니다.

색 대비는 `scratchpad/cvd.py` 로 확인했습니다(최악 쌍 OKLab dE 9.9, 색각 이상 포함 통과).
선마다 점선 모양도 다르게 두어 색만으로 구분하지 않아도 됩니다.
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

HERE = os.path.dirname(os.path.abspath(__file__))

INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
GRID = "#e4e3de"
SURFACE = "#ffffff"

C_FIX = "#eb6834"
C_SUCC = "#1baf7a"
C_RET = "#2a78d6"
C_REF = "#52514e"

D_FIX = (5, 2.2)
D_SUCC = None
D_RET = (2, 1.7)
D_REF = (1.4, 2.4)

KO = dict(
    title="Jump-Start 커리큘럼을 다시 재기",
    sub="제어기가 앞 h 스텝을 끌고 간 뒤 정책이 받습니다 · h 를 내리는 규칙만 다르고 환경·보상·PPO 설정은 전부 동일 · 시드 3개 · 띠 = 시드 최소~최대",
    r1="제어기가 끌고 가는 구간 h", r2="정책 혼자 평가한 리턴 (제어기 없이)",
    y1="h [s]  (에피소드 5초 중)", y2="평가 리턴",
    x="학습 벽시계 시간 [s]",
    c1="예산 60초 — 다른 방식들과 같은 조건", c2="예산 240초 — 아예 못 하는 건지 확인",
    fix="시간 기반 (기존)", succ="성능 기반 · 성공률 게이트", ret="성능 기반 · 리턴 게이트",
    ref="PPO 그대로 (제어기 없이)", solved="풀었다",
    note="성능 기반은 지금 h 에서 제어기+정책 합성 성적이 기준을 넘을 때만 h 를 한 단계(0.5초) 내립니다 — 성공률 게이트는 성공률 ≥ 0.8, 리턴 게이트는 제어기 혼자 리턴의 0.9배(268) · 제어기가 끄는 스텝도 물리 연산이라 벽시계에 들어갑니다 · 시간 기반의 h 선은 당시 로그에 h 가 빠져 있어 그 일정식으로 다시 그린 것입니다(시드 무관)",
)
EN = dict(
    title="Re-measuring the Jump-Start curriculum",
    sub="the guide drives the first h steps, then the policy takes over | only the rule for lowering h differs | 3 seeds",
    r1="guided prefix h", r2="return of the policy alone (no guide)",
    y1="h [s] (of a 5 s episode)", y2="evaluation return",
    x="training wall-clock time [s]",
    c1="60 s budget - same as the other methods", c2="240 s budget - can it get there at all?",
    fix="time-based (before)", succ="performance-based, success gate",
    ret="performance-based, return gate",
    ref="plain PPO (no guide)", solved="solved",
    note="performance-based lowers h one 0.5 s stage only when the guide+policy score clears the bar",
)


def pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic", "Noto Sans KR"]:
        if n in have:
            return n, KO
    return None, EN


def load(d, mode="jsrl"):
    if not d or not os.path.isdir(d):
        return []
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "%s_s*.json" % mode)))]


def series(recs, grid, key, scale=1.0):
    cs = []
    for r in recs:
        t = np.array([p["t"] for p in r["points"]], dtype=float)
        v = np.array([p.get(key, np.nan) for p in r["points"]], dtype=float) * scale
        idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
        c = v[idx]
        c[grid < t[0]] = np.nan
        c[grid > t[-1]] = np.nan
        cs.append(c)
    return np.array(cs)


def band(ax, grid, recs, key, col, lab, scale=1.0, dash=None, step=False, lw=2.3):
    if not recs:
        return
    c = series(recs, grid, key, scale)
    ax.fill_between(grid, np.nanmin(c, axis=0), np.nanmax(c, axis=0),
                    color=col, alpha=0.12, lw=0, zorder=2)
    kw = dict(color=col, lw=lw, zorder=3, label=lab, solid_capstyle="round")
    if step:
        kw["drawstyle"] = "steps-post"
    ln, = ax.plot(grid, np.nanmean(c, axis=0), **kw)
    if dash:
        ln.set_dashes(dash)


def mark_solve(ax, recs, grid, col):
    """처음 성공한 지점에 점과 시간을 직접 적습니다."""
    ok = [r["t_solve"] for r in recs if r["t_solve"] is not None]
    if not ok or len(ok) < len(recs):
        return
    t = float(np.mean(ok))
    c = series(recs, grid, "ret")
    y = float(np.nanmean(c, axis=0)[int(np.clip(np.searchsorted(grid, t), 0, len(grid) - 1))])
    ax.plot([t], [y], marker="o", ms=7.5, mfc=SURFACE, mec=col, mew=2.0, zorder=6)
    ax.annotate("%.0f s" % t, (t, y), xytext=(9, -3), textcoords="offset points",
                ha="left", va="top", fontsize=10, fontweight="bold", color=col)


def frame(ax, xlab=None):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if xlab:
        ax.set_xlabel(xlab, fontsize=10.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixed", default=os.path.join(HERE, "runs_prior_jsrl_fixed"))
    p.add_argument("--succ", default=os.path.join(HERE, "runs_prior"))
    p.add_argument("--ret", default=os.path.join(HERE, "runs_prior_jsrl_ret"))
    p.add_argument("--succ-long", default=os.path.join(HERE, "runs_prior_jsrl_long"))
    p.add_argument("--ret-long", default=os.path.join(HERE, "runs_prior_jsrl_ret_long"))
    p.add_argument("--out", default=os.path.join(HERE, "figures"))
    p.add_argument("--name", default="furuta_rl_jsrl")
    a = p.parse_args()

    font, L = pick_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK3, "ytick.color": INK3, "font.size": 10,
    })

    fixed = load(a.fixed)
    succ = load(a.succ)
    ret = load(a.ret)
    succ_l = load(a.succ_long)
    ret_l = load(a.ret_long)
    scratch = load(a.succ, "scratch")
    if not succ:
        print("no gated runs in", a.succ)
        return

    dt = 0.02
    short = [(fixed, C_FIX, L["fix"], D_FIX), (succ, C_SUCC, L["succ"], D_SUCC),
             (ret, C_RET, L["ret"], D_RET)]
    long_ = [(succ_l, C_SUCC, L["succ"], D_SUCC), (ret_l, C_RET, L["ret"], D_RET)]
    short = [s for s in short if s[0]]
    long_ = [s for s in long_ if s[0]]

    b1 = max(r["budget"] for s in short for r in s[0])
    b2 = max([r["budget"] for s in long_ for r in s[0]] or [b1])
    g1 = np.arange(0.0, b1 + 1e-9, 0.25)
    g2 = np.arange(0.0, b2 + 1e-9, 0.5)

    fig = plt.figure(figsize=(14.6, 8.6), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.74, 1.0],
                          left=0.062, right=0.982, top=0.785, bottom=0.10,
                          wspace=0.15, hspace=0.30)
    ax = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]

    if fixed:
        h_fix = np.round(200.0 * np.maximum(0.0, 1.0 - (g1 / b1) / 0.5)) * dt
        ln, = ax[0][0].plot(g1, h_fix, color=C_FIX, lw=2.3, zorder=3,
                            label=L["fix"], drawstyle="steps-post")
        ln.set_dashes(D_FIX)
    for recs, col, lab, dash in short:
        if recs is fixed:
            continue
        band(ax[0][0], g1, recs, "h", col, lab, scale=dt, dash=dash, step=True)
    for recs, col, lab, dash in long_:
        band(ax[0][1], g2, recs, "h", col, lab, scale=dt, dash=dash, step=True)

    for axis, bud in ((ax[0][0], b1), (ax[0][1], b2)):
        frame(axis)
        axis.set_xlim(0, bud)
        axis.set_ylim(-0.3, 5.25)
        axis.set_yticks([0, 1, 2, 3, 4, 5])
        axis.axhline(0.0, color=INK3, lw=1.0, ls=(0, (5, 4)), zorder=1)
    ax[0][0].set_ylabel(L["y1"], fontsize=10.5)
    ax[0][1].tick_params(labelleft=False)
    ax[0][0].legend(loc="upper right", frameon=False, fontsize=9.5,
                    handlelength=2.6, handletextpad=0.7, labelspacing=0.4)

    band(ax[1][0], g1, scratch, "ret", C_REF, L["ref"], dash=D_REF, lw=2.0)
    for recs, col, lab, dash in short:
        band(ax[1][0], g1, recs, "ret", col, lab, dash=dash)
    band(ax[1][1], g2, scratch, "ret", C_REF, L["ref"], dash=D_REF, lw=2.0)
    for recs, col, lab, dash in long_:
        band(ax[1][1], g2, recs, "ret", col, lab, dash=dash)

    mark_solve(ax[1][0], scratch, g1, C_REF)
    for recs, col, _lab, _d in short:
        mark_solve(ax[1][0], recs, g1, col)
    for recs, col, _lab, _d in long_:
        mark_solve(ax[1][1], recs, g2, col)

    for axis, bud in ((ax[1][0], b1), (ax[1][1], b2)):
        frame(axis, L["x"])
        axis.set_xlim(0, bud)
        axis.set_ylim(-540.0, 360.0)
    ax[1][0].set_ylabel(L["y2"], fontsize=10.5)
    ax[1][1].tick_params(labelleft=False)
    ax[1][0].legend(loc="upper left", frameon=False, fontsize=9.5,
                    handlelength=2.6, handletextpad=0.7, labelspacing=0.4)

    for c, lab in ((0, L["c1"]), (1, L["c2"])):
        ax[0][c].set_title(lab, fontsize=12, color=INK, loc="left", pad=24,
                           fontweight="bold")
    for r, lab in ((0, L["r1"]), (1, L["r2"])):
        for c in (0, 1):
            ax[r][c].annotate(lab, (0.0, 1.0), xycoords="axes fraction",
                              xytext=(0, 6), textcoords="offset points",
                              ha="left", va="bottom", fontsize=10.5, color=INK2)

    fig.suptitle(L["title"], x=0.062, ha="left", fontsize=15.5, fontweight="bold", y=0.965)
    fig.text(0.062, 0.905, L["sub"], ha="left", fontsize=9.5, color=INK2)
    fig.text(0.062, 0.018, L["note"], ha="left", fontsize=8.2, color=INK3, wrap=True)

    os.makedirs(a.out, exist_ok=True)
    png = os.path.join(a.out, a.name + ".png")
    fig.savefig(png, dpi=150)
    fig.savefig(os.path.join(a.out, a.name + ".svg"))
    plt.close(fig)
    print("wrote", png, "(font: %s)" % (font or "default/EN"))

    print("%-30s %7s %9s %9s %9s %8s %9s %9s" % (
        "set", "budget", "solve", "stage", "h_end[s]", "fin_sc", "fin_ret", "steps/s"))
    rows = [("시간 기반", fixed), ("성능 기반 · 성공률", succ), ("성능 기반 · 리턴", ret),
            ("성능 기반 · 성공률 (240s)", succ_l), ("성능 기반 · 리턴 (240s)", ret_l),
            ("PPO 그대로", scratch)]
    for lab, recs in rows:
        if not recs:
            continue
        ok = [r["t_solve"] for r in recs if r["t_solve"] is not None]
        print("%-30s %7.0f %9s %9s %9.2f %8.2f %9.0f %9.0f" % (
            lab, recs[0]["budget"],
            ("%.0fs" % np.mean(ok)) if len(ok) == len(recs) else "%d/%d" % (len(ok), len(recs)),
            "%.1f/%d" % (np.mean([r.get("stage", 0) for r in recs]), recs[0].get("stages", 0)),
            np.mean([r.get("h_end", 0) for r in recs]) * dt,
            np.mean([r["final_succ"] for r in recs]),
            np.mean([r["final_ret"] for r in recs]),
            np.mean([r["steps"] / r["train_time"] for r in recs])))


if __name__ == "__main__":
    main()
