"""Plot the saved experimental results for the project report."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]


def configure_font():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    families = ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "STHeiti", "PingFang HK")
    font = next((name for name in families if name in installed), None)
    if font is None:
        raise RuntimeError("Install Noto Sans CJK SC to render the Chinese labels")
    plt.rcParams.update({"font.family": font, "font.size": 10.5, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "legend.frameon": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    configure_font()
    table = pd.read_csv(ROOT / "results/inventory_replay/case_pack_by_fold.csv")
    table["unit_cost"] = table.cost / table.weeks_items
    paired = table.pivot(index=["fold", "lead", "forecast"], columns="case_pack", values="unit_cost")
    change = (paired[12] - paired[1]).unstack(["lead", "forecast"])
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8.5, 3.7), layout="constrained")
    for lead, ax in enumerate(axes, start=1):
        ax.axhline(0, color=".35", lw=.9)
        for model, label, marker, color, offset in (
            ("naive", "上周销量", "o", "#476b83", -.10), ("q50", "中位数预测", "^", "#a45735", .10)
        ):
            ax.scatter(change.index + offset, change[lead, model], label=label, marker=marker,
                       color=color, s=33, zorder=3)
        ax.set(title=f"提前期 {lead} 周", xlabel="回测折次", xticks=change.index,
               xlim=(.6, len(change)+.4), ylim=(-.23, .026))
        ax.grid(axis="y", color=".88", lw=.6)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[0].set_ylabel("12 件箱规减逐件补货费用\n（美元/商品门店周）")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.savefig(ROOT.parent / "images/case_pack.png", dpi=200)

    plt.close(fig)


if __name__ == "__main__":
    main()
