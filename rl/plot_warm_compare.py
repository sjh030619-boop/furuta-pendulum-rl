"""배우(정책)만 들고 가면 어떻게 되는가 — off-policy와 on-policy를 나란히.

    python rl/plot_warm_compare.py

같은 축 위에 올려야 보입니다. SAC는 난수 비평가가 멀쩡한 정책을 끌어내려 2초 만에 무너지고,
PPO는 같은 조건에서 꿈쩍도 하지 않습니다. 값함수가 쓰이는 방식이 다르기 때문입니다.

  SAC  비평가 Q(s,a)가 "이 행동의 가치는 얼마"를 직접 가르칩니다. 난수면 틀린 방향을 가르칩니다.
  PPO  값함수 V(s)는 실제로 받은 보상에서 빼는 **기준선**입니다. 난수여도 이점 추정이 시끄러워질
       뿐 부호가 뒤집히지 않고, clip이 한 번에 크게 바뀌는 것을 막습니다.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plot_warmstart import (load, curves, pick_font, INK, INK2, INK3, GRID, SURFACE)

KEYS = [
    ("scratch", "#8a8880", (None, None), "맨 처음부터", "from scratch", 3),
    ("actor", "#e34948", (1.4, 1.4), "배우만 들고", "actor only", 5),
    ("full", "#2a78d6", (None, None), "배우+비평가 들고", "actor + critic", 4),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sac", default=os.path.join(HERE, "runs_warm"))
    p.add_argument("--ppo", default=os.path.join(HERE, "runs_warm_ppo"))
    p.add_argument("--out", default=os.path.join(HERE, "figures"))
    p.add_argument("--name", default="furuta_rl_warm_compare")
    a = p.parse_args()

    font, _L = pick_font()
    ko = font is not None
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK3, "ytick.color": INK3, "font.size": 10,
    })

    sets = [("SAC", "off-policy", load(a.sac)), ("PPO", "on-policy", load(a.ppo))]
    budget = max(r["budget"] for _n, _f, d in sets for v in d.values() for r in v)
    grid = np.arange(0.0, budget + 1e-9, 0.25)

    fig = plt.figure(figsize=(12.6, 6.1), dpi=150)
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.982, top=0.745, bottom=0.145, wspace=0.10)
    axes = []
    for i, (name, fam, data) in enumerate(sets):
        ax = fig.add_subplot(gs[0, i], sharey=axes[0] if axes else None)
        for key, col, dash, lk, lе, z in KEYS:
            if key not in data:
                continue
            c = curves(data[key], grid)
            ax.fill_between(grid, np.nanmin(c, axis=0), np.nanmax(c, axis=0),
                            color=col, alpha=0.13, lw=0, zorder=2)
            ln, = ax.plot(grid, np.nanmean(c, axis=0), color=col, lw=2.4, zorder=z,
                          label=(lk if ko else lе), solid_capstyle="round")
            if dash[0]:
                ln.set_dashes(dash)
        ax.set_title("%s  (%s)" % (name, fam), fontsize=12.5, color=INK,
                     loc="left", pad=10, fontweight="bold")
        ax.set_xlabel("학습 벽시계 시간 [s]" if ko else "training wall-clock time [s]", fontsize=10.5)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlim(0, budget)
        ax.legend(loc="lower right", frameon=False, fontsize=10, labelspacing=0.5,
                  handlelength=2.6, handletextpad=0.7)
        if i:
            ax.tick_params(labelleft=False)
        axes.append(ax)

    axes[0].set_ylabel("평가 리턴 (고정 초기조건 5개 평균)" if ko
                       else "evaluation return (mean of 5 fixed starts)", fontsize=10.5)
    allv = np.concatenate([curves(v, grid).ravel() for _n, _f, d in sets for v in d.values()
                           if len(v)])
    allv = allv[~np.isnan(allv)]
    axes[0].set_ylim(np.percentile(allv, 1.0), max(allv) * 1.06)

    if ko:
        axes[0].annotate("\n".join(["난수 비평가가", "멀쩡한 정책을 끌어내립니다"]),
                         xy=(3.0, -150), xytext=(7.5, 90), fontsize=10, color="#b3302f",
                         linespacing=1.45, zorder=6,
                         arrowprops=dict(arrowstyle="->", color="#b3302f", lw=1.3,
                                         shrinkA=3, shrinkB=3,
                                         connectionstyle="arc3,rad=-0.25"))
        axes[1].annotate("\n".join(["같은 조건인데", "꿈쩍도 하지 않습니다"]),
                         xy=(14, 299), xytext=(15, 150), fontsize=10, color="#b3302f",
                         linespacing=1.45, zorder=6,
                         arrowprops=dict(arrowstyle="->", color="#b3302f", lw=1.3,
                                         shrinkA=3, shrinkB=5,
                                         connectionstyle="arc3,rad=0.25"))

    fig.suptitle("배우(정책)만 들고 가면 어떻게 되는가" if ko
                 else "What happens if you carry only the actor",
                 x=0.075, ha="left", fontsize=15.5, fontweight="bold", y=0.955)
    fig.text(0.075, 0.875,
             ("같은 과제에서 180초 학습한 정책에서 출발 · 시드 3개 · 띠 = 시드 최소~최대 · 평가 시간은 시계에서 제외"
              if ko else "source: a policy trained 180 s on the same task | 3 seeds | band = seed min-max"),
             ha="left", fontsize=10, color=INK2)
    fig.text(0.075, 0.030,
             ("두 알고리즘의 리턴 값은 같은 보상으로 잰 것이라 서로 비교해도 됩니다" if ko
              else "both panels use the same reward, so the values are comparable"),
             ha="left", fontsize=8.5, color=INK3)

    os.makedirs(a.out, exist_ok=True)
    png = os.path.join(a.out, a.name + ".png")
    fig.savefig(png, dpi=150)
    fig.savefig(os.path.join(a.out, a.name + ".svg"))
    plt.close(fig)
    print("wrote", png)

    print("%-5s %-9s %8s %9s %9s" % ("algo", "mode", "t0_ret", "worst", "fin_ret"))
    for name, _f, data in sets:
        for key, _c, _d, _lk, _le, _z in KEYS:
            if key not in data:
                continue
            v = data[key]
            c = curves(v, grid)
            print("%-5s %-9s %8.0f %9.0f %9.0f" % (
                name, key, np.mean([r["ret0"] for r in v]),
                float(np.nanmin(np.nanmean(c, axis=0))),
                np.mean([r["final_ret"] for r in v])))


if __name__ == "__main__":
    main()
