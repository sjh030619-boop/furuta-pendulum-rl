"""학습된 정책을 들고 시작했을 때 초반 학습이 어떻게 달라지는지 그립니다.

    python rl/plot_warmstart.py

왼쪽: 같은 규칙으로 이어서 학습할 때. 무엇을 들고 가느냐(배우만 / 배우+비평가 / 거기에 버퍼까지)로
나눠 그립니다. 오른쪽: 규칙이 바뀐 뒤 — 교육용 키트에서 실제로 일어나는 경우입니다.
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

STYLE = {
    "scratch":    ("#8a8880", (None, None), "맨 처음부터", "from scratch"),
    "actor":      ("#e34948", (1.4, 1.4), "배우만 들고", "actor only"),
    "full":       ("#2a78d6", (None, None), "배우+비평가", "actor + critic"),
    "full_buf":   ("#1baf7a", (5, 1.6), "배우+비평가+버퍼", "+ prefilled buffer"),
    "scratch_r1": ("#8a8880", (None, None), "맨 처음부터", "from scratch"),
    "full_r1":    ("#6250d6", (None, None), "들고 시작", "warm start"),
    "scratch_r2": ("#8a8880", (None, None), "맨 처음부터", "from scratch"),
    "full_r2":    ("#6250d6", (None, None), "들고 시작", "warm start"),
}
PANELS = [
    (["scratch", "actor", "full", "full_buf"], "p1"),
    (["scratch_r1", "full_r1"], "p2"),
    (["scratch_r2", "full_r2"], "p3"),
]

KO = dict(
    title="학습된 정책을 들고 시작하면 초반 학습이 나아지는가",
    sub="%s · 출발점은 같은 과제에서 180초 학습한 정책 · 시드 3개 · 띠 = 시드 최소~최대 · 평가 시간은 시계에서 제외",
    p1="같은 규칙으로 이어서 학습",
    p2="규칙이 조금 바뀐 뒤\n(힘 아끼기 약하게 → 강하게)",
    p3="규칙이 많이 바뀐 뒤\n(부드럽게 약하게 → 강하게)",
    x="학습 벽시계 시간 [s]", y="평가 리턴 (고정 초기조건 5개 평균)",
    note="버퍼를 채우는 시간도 학습 시계에 넣었습니다 · 리턴은 각 패널이 쓰는 보상 규칙으로 잰 값이라 두 패널의 값은 서로 비교하지 마세요",
)
EN = dict(
    title="Does starting from a trained policy improve early learning?",
    sub="%s | source: a policy trained 180 s on the same task | 3 seeds | band = seed min-max",
    p1="continuing with the same reward", p2="after a small rule change",
    p3="after a large rule change",
    x="training wall-clock time [s]", y="evaluation return (mean of 5 fixed starts)",
    note="buffer prefill time is counted against the training clock",
)




ANNO = "\n".join(["비평가가 난수라서", "2초 만에 무너집니다", "(되돌아오는 데 30초)"])


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
        v = np.array([p[key] for p in r["points"]])
        idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
        c = v[idx].astype(float)
        c[grid < t[0]] = np.nan
        cs.append(c)
    return np.array(cs)


def panel(ax, data, keys, grid, L, font, lo_hint=None):
    for k in keys:
        if k not in data:
            continue
        col, dash, ko, en = STYLE[k]
        c = curves(data[k], grid)
        m = np.nanmean(c, axis=0)
        ax.fill_between(grid, np.nanmin(c, axis=0), np.nanmax(c, axis=0),
                        color=col, alpha=0.13, lw=0, zorder=2)
        ln, = ax.plot(grid, m, color=col, lw=2.3, zorder=3,
                      label=(ko if font else en), solid_capstyle="round")
        if dash[0]:
            ln.set_dashes(dash)
    ax.set_xlabel(L["x"], fontsize=10.5)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlim(0, grid[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(HERE, "runs_warm"))
    p.add_argument("--out", default=os.path.join(HERE, "figures"))
    p.add_argument("--name", default="furuta_rl_warmstart")
    p.add_argument("--algo", default="SAC")
    p.add_argument("--anno", action="store_true", help="SAC 전용 설명 화살표를 넣습니다")
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
    budget = max(r["budget"] for v in data.values() for r in v)
    grid = np.arange(0.0, budget + 1e-9, 0.25)

    panels = [(keys, tk) for keys, tk in PANELS if any(k in data for k in keys)]
    n = len(panels)
    fig = plt.figure(figsize=(5.0 * n + 1.3, 6.5), dpi=150)
    gs = fig.add_gridspec(1, n, left=0.062 if n > 1 else 0.10, right=0.982,
                          top=0.745, bottom=0.135, wspace=0.13)
    axes = []
    for i, (keys, tk) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i], sharey=axes[0] if axes else None)
        panel(ax, data, keys, grid, L, font)
        ax.set_title(L[tk], fontsize=11.5, color=INK, loc="left", pad=10,
                     fontweight="bold", linespacing=1.4)
        ax.legend(loc="lower right", frameon=False, fontsize=10, labelspacing=0.5,
                  handlelength=2.6, handletextpad=0.7)
        if i:
            ax.tick_params(labelleft=False)
        axes.append(ax)
    axes[0].set_ylabel(L["y"], fontsize=10.5)

    allv = np.concatenate([curves(v, grid).ravel() for v in data.values()])
    allv = allv[~np.isnan(allv)]
    axes[0].set_ylim(np.percentile(allv, 1.5), max(allv) * 1.06)

    if a.anno and font:
        axes[0].annotate(ANNO,
                         xy=(2.6, -160), xytext=(6.0, 55), fontsize=9.5, color="#b3302f",
                         linespacing=1.45, zorder=6,
                         arrowprops=dict(arrowstyle="->", color="#b3302f", lw=1.3,
                                         shrinkA=3, shrinkB=3, connectionstyle="arc3,rad=-0.25"))

    fig.suptitle(L["title"] + "  —  " + a.algo, x=0.062, ha="left",
                 fontsize=15.5, fontweight="bold", y=0.962)
    fig.text(0.062, 0.885, L["sub"] % a.algo, ha="left", fontsize=10, color=INK2)
    fig.text(0.062, 0.028, L["note"], ha="left", fontsize=8.5, color=INK3)

    os.makedirs(a.out, exist_ok=True)
    png = os.path.join(a.out, a.name + ".png")
    fig.savefig(png, dpi=150)
    fig.savefig(os.path.join(a.out, a.name + ".svg"))
    plt.close(fig)
    print("wrote", png, "(font: %s)" % (font or "default/EN"))

    print("%-11s %7s %7s %9s %8s %8s %8s" % ("mode", "t0succ", "t0ret", "solve", "fin_sc", "fin_ret", "worst"))
    for keys, _tk in PANELS:
        for k in keys:
            if k not in data:
                continue
            v = data[k]
            ok = [r["t_solve"] for r in v if r["t_solve"] is not None]
            c = curves(v, grid)
            worst = float(np.nanmin(np.nanmean(c, axis=0)))
            print("%-11s %7.2f %7.0f %9s %8.2f %8.0f %8.0f" % (
                k, np.mean([r["succ0"] for r in v]), np.mean([r["ret0"] for r in v]),
                ("%.0fs" % np.mean(ok)) if len(ok) == len(v) else ("%d/%d 성공" % (len(ok), len(v))),
                np.mean([r["final_succ"] for r in v]), np.mean([r["final_ret"] for r in v]), worst))


if __name__ == "__main__":
    main()
