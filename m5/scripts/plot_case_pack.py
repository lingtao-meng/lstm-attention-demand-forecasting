"""Plot the paired cost differences for both case sizes in each M5 fold."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
INK, GRAY = "#1b2c38", "#4e606d"
BLUE, ORANGE = "#236b8e", "#a64b30"



def configure_style():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "STHeiti", "PingFang HK")
    family = next((name for name in candidates if name in installed), None)
    if family is None:
        raise RuntimeError("Install Noto Sans CJK SC to render the Chinese labels")
    plt.rcParams.update({
        "font.family": family, "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": "#aebac2", "axes.linewidth": .75,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": GRAY, "ytick.color": GRAY, "axes.unicode_minus": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "grid.color": "#e5eaee", "grid.linewidth": .7,
    })


def main():
    configure_style()
    table = pd.read_csv(ROOT / "results/inventory_replay/case_pack_by_fold.csv")
    unit_cost = table.assign(unit_cost=table.cost/table.weeks_items)
    paired = unit_cost.pivot(index=["lead", "forecast", "fold"], columns="case_pack", values="unit_cost")
    difference = paired[[6, 12]].sub(paired[1], axis=0)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8.8, 5.4))
    fig.subplots_adjust(left=.135, right=.94, bottom=.15, top=.82, hspace=.35, wspace=.16)
    for row, (forecast, label) in enumerate((("naive", "上周销量"), ("q50", "中位数预测"))):
        for col, lead in enumerate((1, 2)):
            ax = axes[row, col]
            values = difference.loc[lead, forecast].sort_index()
            folds = values.index.to_numpy()
            ax.axhspan(0, .032, facecolor="#fbf0eb", linewidth=0)
            ax.axhline(0, color="#52636e", linewidth=1.0)
            ax.vlines(folds, values[6], values[12], color="#aebbc3", linewidth=1.3)
            ax.scatter(folds, values[6], color=BLUE, s=33, zorder=3)
            ax.scatter(folds, values[12], color=ORANGE, s=36, marker="D", zorder=4)
            ax.set(xlim=(.55, 6.45), ylim=(-.224, .032), xticks=folds,
                   yticks=[-.20, -.15, -.10, -.05, 0])
            ax.set_axisbelow(True)
            ax.grid(axis="y")
            ax.tick_params(length=2.5)
            if row == 0:
                ax.set_title(f"提前期 {lead} 周", pad=8)
            if col == 1:
                ax.text(1.035, .5, label, transform=ax.transAxes, va="center", rotation=-90)
    fig.supxlabel("回测折次", y=.050, fontsize=11, color=INK)
    fig.supylabel("相对逐件补货的费用差（美元/商品门店周）", x=.025, fontsize=11, color=INK)
    handles = [
        Line2D([], [], color=BLUE, marker="o", ls="", label="6 件一箱"),
        Line2D([], [], color=ORANGE, marker="D", ls="", label="12 件一箱"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .982), ncol=2, fontsize=10)
    fig.savefig(ROOT.parent / "images/case_pack.png", dpi=210)
    plt.close(fig)


if __name__ == "__main__":
    main()
