"""SAC 벽시계 조율 결과를 한 장으로.

    python rl/plot_speed.py                 # 세 칸 전부
    python rl/plot_speed.py --confirm none  # 확정 실험 없이 왼쪽 칸만

왼쪽: 설정을 하나씩 바꿨을 때 풀린 시각. 점 = 개별 시드, 마름모 = 평균, 회색 띠 = 기본
      설정의 시드 범위. 모든 시드가 그 띠 밖에 있어야 "달라졌다"고 보고, 하나라도 겹치면
      회색으로 둡니다.
오른쪽 위: 학습률만 바꿨을 때. 어디서 평평해지는지를 봅니다.
오른쪽 아래: 확정 실험(일찍 멈추지 않고 끝까지 돌린 것)의 성공률 곡선.

끝에 확정 실험의 안정성 지표를 벤치마크 성적표와 같은 정의로 찍습니다
(`plot_results.metrics`: 최종 = 마지막 20% 구간, 시드 편차 = 시드 간 성공률 최대-최소의
시간 평균, 낙폭 = 시드 평균 리턴이 그때까지의 최고치에서 떨어진 최댓값).

색은 `scratchpad/cvd.py` 로 확인한 조합입니다(색각 이상 포함 최악 쌍 OKLab dE 9.9).
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

INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
GRID = "#e4e3de"
SURFACE = "#ffffff"
C_FAST = "#2a78d6"
C_SLOW = "#eb6834"
C_SAME = "#8a8880"
CURVE = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a"]
DASH = [(5, 2.2), None, (2, 1.7), (6, 1.6, 1.4, 1.6)]

ONE_AT_A_TIME = ["base", "b512", "b1024", "b2048", "lr1e-3", "lr3e-3", "n4", "n16", "n32",
                 "g2", "tau02", "ls500", "ent01", "nstep3", "ln", "qf256"]
LR_SWEEP = [("base", 3e-4), ("lr6e-4", 6e-4), ("lr1e-3", 1e-3), ("lr1.5e-3", 1.5e-3),
            ("lr2e-3", 2e-3), ("lr3e-3", 3e-3)]

LABEL = {
    "base": "기본 설정",
    "b512": "배치 512",
    "b1024": "배치 1024",
    "b2048": "배치 2048",
    "lr1e-3": "학습률 1e-3",
    "lr2e-3": "학습률 2e-3",
    "lr3e-3": "학습률 3e-3",
    "n4": "환경 4개",
    "n16": "환경 16개",
    "n32": "환경 32개",
    "g2": "스텝당 업데이트 2회",
    "tau02": "타깃 갱신 τ 0.02",
    "ls500": "학습 시작 500스텝",
    "ent01": "엔트로피 계수 0.1에서 시작",
    "nstep3": "3-step 리턴",
    "ln": "비평가 LayerNorm",
    "qf256": "비평가 256×256",
}
CURVE_LABEL = {"base": "기본 (학습률 3e-4)", "lr1e-3": "학습률 1e-3", "lr2e-3": "학습률 2e-3"}


def pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic", "Noto Sans KR"]:
        if n in have:
            return n
    return None


def load(d):
    rows = {}
    if not d or not os.path.isdir(d):
        return rows
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        rows.setdefault(r["variant"], []).append(r)
    return rows


def curve(recs, grid, key="success"):
    cs = []
    for r in recs:
        t = np.array([p["t"] for p in r["points"]], dtype=float)
        v = np.array([p[key] for p in r["points"]], dtype=float)
        idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
        cs.append(v[idx])
    return np.array(cs)


def stability(recs, budget):
    """벤치마크 성적표와 같은 정의."""
    grid = np.arange(0.0, budget + 1e-9, 0.5)
    succ = curve(recs, grid, "success")
    ret = curve(recs, grid, "ret")
    tail = grid >= grid[-1] * 0.8
    mret = ret.mean(axis=0)
    runmax = np.maximum.accumulate(mret)
    drop = runmax - mret
    span = runmax[-1] - mret[0]
    worst = float(min(succ[:, grid >= grid[-1] * 0.3].min(axis=1)))
    ts = [r["t_sust"] for r in recs]
    return dict(
        t_sust=float(np.mean(ts)) if all(t is not None for t in ts) else None,
        t_each=ts,
        final_success=float(succ[:, tail].mean()),
        final_ret=float(ret[:, tail].mean()),
        seed_spread=float(np.mean(succ.max(axis=0) - succ.min(axis=0))),
        drawdown=float(drop.max()),
        drawdown_pct=float(drop.max() / span * 100.0) if span > 50.0 else float("nan"),
        worst_late=worst,
        fps=float(np.mean([r["fps"] for r in recs])),
        calib=max(max(r["calib_ms_before"], r["calib_ms_after"]) for r in recs),
    )


def frame(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(HERE, "runs_speed"))
    p.add_argument("--confirm", default=os.path.join(HERE, "runs_speed_confirm"))
    p.add_argument("--finals", nargs="*", default=["base", "lr1e-3", "lr2e-3"])
    p.add_argument("--out", default=os.path.join(HERE, "figures"))
    p.add_argument("--name", default="furuta_rl_sac_speed")
    p.add_argument("--curve-xmax", type=float, default=0.0)
    a = p.parse_args()

    font = pick_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK3, "ytick.color": INK3, "font.size": 10,
    })

    allrows = load(a.runs)
    rows = {k: allrows[k] for k in ONE_AT_A_TIME if k in allrows}
    if "base" not in rows:
        print("기본 설정 결과가 없습니다:", a.runs)
        return
    budget = max(r["budget"] for v in rows.values() for r in v)
    conf = load(a.confirm)
    finals = [k for k in a.finals if k in conf][:4]
    lr_rows = [(k, lr) for k, lr in LR_SWEEP if k in allrows]
    wide = bool(finals) or len(lr_rows) >= 3

    def mean_t(v):
        ok = [r["t_sust"] for r in v if r["t_sust"] is not None]
        return float(np.mean(ok)) if len(ok) == len(v) else None

    base_t = [r["t_sust"] for r in rows["base"] if r["t_sust"] is not None]
    b_lo, b_hi, b_mean = min(base_t), max(base_t), float(np.mean(base_t))
    order = sorted(rows, key=lambda k: (mean_t(rows[k]) is None,
                                        mean_t(rows[k]) if mean_t(rows[k]) is not None else 0.0))

    h = 8.6
    fig = plt.figure(figsize=(15.2 if wide else 9.6, h), dpi=150)
    top = 1.0 - 1.25 / h
    bot = 1.0 / h
    if wide:
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.92], height_ratios=[0.8, 1.0],
                              left=0.16, right=0.975, top=top, bottom=bot,
                              wspace=0.40, hspace=0.42)
        ax = fig.add_subplot(gs[:, 0])
        axl = fig.add_subplot(gs[0, 1])
        axc = fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(1, 1, left=0.25, right=0.86, top=top, bottom=bot)
        ax = fig.add_subplot(gs[0, 0])
        axl = axc = None

    ys = np.arange(len(order))[::-1]
    ax.axvspan(b_lo, b_hi, color=INK3, alpha=0.14, lw=0, zorder=1)
    ax.axvline(b_mean, color=INK2, lw=1.2, ls=(0, (5, 4)), zorder=2)
    xmax = min(budget, max(r["t_sust"] for v in rows.values() for r in v
                           if r["t_sust"] is not None) * 1.18)
    for y, k in zip(ys, order):
        v = sorted(rows[k], key=lambda r: r["seed"])
        ts = [r["t_sust"] for r in v]
        ok = [t for t in ts if t is not None]
        m = mean_t(v)
        if k == "base":
            col = INK2
        elif len(ok) < len(ts) or min(ok) > b_hi:
            col = C_SLOW
        elif max(ok) < b_lo:
            col = C_FAST
        else:
            col = C_SAME
        if ok:
            ax.plot([min(ok), max(ok)], [y, y], color=col, lw=2.0, alpha=0.55, zorder=3,
                    solid_capstyle="round")
            ax.plot(ok, [y] * len(ok), ls="none", marker="o", ms=5.5, mfc=SURFACE, mec=col,
                    mew=1.4, zorder=4)
        miss = len(ts) - len(ok)
        if miss:
            ax.plot([xmax * 0.985] * miss, [y] * miss, ls="none", marker="x", ms=7,
                    mec=C_SLOW, mew=1.8, zorder=4, clip_on=False)
        if m is not None:
            ax.plot([m], [y], ls="none", marker="D", ms=8, mfc=col, mec=SURFACE, mew=1.2, zorder=5)
            txt = "%.1f s" % m
            if k != "base":
                txt += "  (%+.0f%%)" % (100.0 * (m / b_mean - 1.0))
        else:
            txt = "%d/%d만 풀림" % (len(ok), len(ts))
        ax.annotate(txt, (1.0, y), xycoords=("axes fraction", "data"), xytext=(8, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=9.5,
                    color=INK, fontweight="bold" if k == "base" else "normal",
                    annotation_clip=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL.get(k, k) for k in order], fontsize=10.5)
    for lab, k in zip(ax.get_yticklabels(), order):
        if k == "base":
            lab.set_fontweight("bold")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("풀린 시각 [s]", fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Line2D([], [], ls="none", marker="D", ms=7, mfc=C_FAST, mec=SURFACE,
               label="모든 시드가 기본보다 빠름"),
        Line2D([], [], ls="none", marker="D", ms=7, mfc=C_SAME, mec=SURFACE,
               label="기본과 시드 범위가 겹침"),
        Line2D([], [], ls="none", marker="D", ms=7, mfc=C_SLOW, mec=SURFACE,
               label="모든 시드가 기본보다 느림"),
        Line2D([], [], ls="none", marker="o", ms=5.5, mfc=SURFACE, mec=INK2, label="개별 시드"),
    ], loc="upper right", frameon=False, fontsize=9, labelspacing=0.35, handletextpad=0.4)
    ax.set_title("설정을 하나씩 바꿨을 때", fontsize=11.5, color=INK, loc="left", pad=10,
                 fontweight="bold")

    if axl is not None:
        xs = [lr for _k, lr in lr_rows]
        ms, tops = [], []
        for k, lr in lr_rows:
            ts = [r["t_sust"] for r in allrows[k] if r["t_sust"] is not None]
            axl.plot([lr] * len(ts), ts, ls="none", marker="o", ms=5.5, mfc=SURFACE,
                     mec=C_FAST if k != "base" else INK2, mew=1.4, zorder=4)
            ms.append(float(np.mean(ts)))
            tops.append(max(ts))
        axl.plot(xs, ms, color=C_FAST, lw=2.3, zorder=3, solid_capstyle="round")
        axl.plot(xs, ms, ls="none", marker="D", ms=7.5, mfc=C_FAST, mec=SURFACE, mew=1.2, zorder=5)
        axl.plot([xs[0]], [ms[0]], ls="none", marker="D", ms=7.5, mfc=INK2, mec=SURFACE,
                 mew=1.2, zorder=6)
        for x, m, t in zip(xs, ms, tops):
            axl.annotate("%.1f s" % m, (x, t), xytext=(0, 9), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9.5, color=INK)
        axl.set_xscale("log")
        axl.set_xticks(xs)
        axl.set_xticklabels(["3e-4\n(기본)", "6e-4", "1e-3", "1.5e-3", "2e-3", "3e-3"][:len(xs)])
        axl.minorticks_off()
        axl.set_ylim(0, max(max(ms) * 1.35, 20))
        axl.set_xlabel("학습률", fontsize=10)
        axl.set_ylabel("풀린 시각 [s]", fontsize=10)
        frame(axl)
        axl.set_title("학습률만 바꿨을 때", fontsize=11.5, color=INK, loc="left", pad=10,
                      fontweight="bold")

    if axc is not None and finals:
        b2 = max(r["budget"] for k in finals for r in conf[k])
        last_bad = 0.0
        for k in finals:
            for r in conf[k]:
                bad = [q["t"] for q in r["points"] if q["success"] < 1.0]
                last_bad = max(last_bad, max(bad) if bad else 0.0)
        cx = a.curve_xmax or min(b2, max(40.0, 10.0 * np.ceil((last_bad + 6.0) / 10.0)))
        grid = np.arange(0.0, cx + 1e-9, 0.25)
        if cx < b2:
            axc.annotate("%.0f초 이후 %.0f초까지는 세 설정 모두\n모든 시드에서 성공률 1.00 (그래서 앞부분만 확대)"
                         % (np.ceil(last_bad), b2), (0.985, 0.42), xycoords="axes fraction",
                         ha="right", va="center", fontsize=9.5, color=INK2, linespacing=1.5)
        for i, k in enumerate(finals):
            c = curve(conf[k], grid)
            axc.fill_between(grid, c.min(axis=0), c.max(axis=0), color=CURVE[i], alpha=0.12,
                             lw=0, zorder=2)
            ln, = axc.plot(grid, c.mean(axis=0), color=CURVE[i], lw=2.3, zorder=3,
                           label=CURVE_LABEL.get(k, LABEL.get(k, k)), solid_capstyle="round")
            if DASH[i]:
                ln.set_dashes(DASH[i])
        axc.axhline(0.8, color=INK3, lw=1.0, ls=(0, (2, 3)), zorder=1)
        axc.set_xlim(0, cx)
        axc.set_ylim(-0.03, 1.06)
        axc.set_xlabel("학습 벽시계 시간 [s]", fontsize=10)
        axc.set_ylabel("평가 성공률", fontsize=10)
        frame(axc)
        axc.legend(loc="lower right", frameon=False, fontsize=9.5, handlelength=2.6)
        n_c = max(len(conf[k]) for k in finals)
        axc.set_title("끝까지 돌렸을 때 (%d초, 시드 %d개, 띠 = 시드 최소~최대)" % (b2, n_c),
                      fontsize=11.5, color=INK, loc="left", pad=10, fontweight="bold")

    n_seed = max(len(v) for v in rows.values())
    x0 = 0.02
    fig.suptitle("SAC를 벽시계 시간 기준으로 조율하기", x=x0, ha="left", fontsize=15,
                 fontweight="bold", y=1.0 - 0.30 / h)
    fig.text(x0, 1.0 - 0.72 / h,
             "기본 = 벤치마크의 SAC (환경 8개, 배치 256, 64×64, 학습률 3e-4) · 스크리닝은 시드 %d개 · "
             "회색 띠 = 기본 설정의 시드 범위, 점선 = 그 평균" % n_seed,
             ha="left", fontsize=9.5, color=INK2)
    fig.text(x0, 0.2 / h,
             "풀린 시각 = 성공률 0.8 이상을 3번 연속 유지하기 시작한 학습 시간 · 평가는 1초 간격(확정 실험은 2초)이고 "
             "평가 시간은 학습 시계에서 뺍니다",
             ha="left", fontsize=8.5, color=INK3)

    os.makedirs(a.out, exist_ok=True)
    png = os.path.join(a.out, a.name + ".png")
    fig.savefig(png, dpi=150)
    fig.savefig(os.path.join(a.out, a.name + ".svg"))
    plt.close(fig)
    print("wrote", png)

    if finals:
        print()
        print("%-10s %7s %14s %9s %9s %9s %9s %10s %7s %6s" % (
            "설정", "풀림", "시드별", "최종성공", "최종리턴", "시드편차", "낙폭%", "후반최저", "fps", "calib"))
        for k in finals:
            s = stability(conf[k], max(r["budget"] for r in conf[k]))
            print("%-10s %7s %14s %9.2f %9.0f %9.3f %9.0f %10.2f %7.0f %6.0f" % (
                k, "-" if s["t_sust"] is None else "%.1f" % s["t_sust"],
                "/".join("x" if t is None else "%.0f" % t for t in s["t_each"]),
                s["final_success"], s["final_ret"], s["seed_spread"], s["drawdown_pct"],
                s["worst_late"], s["fps"], s["calib"]))


if __name__ == "__main__":
    main()
