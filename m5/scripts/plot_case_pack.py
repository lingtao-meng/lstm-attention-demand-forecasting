"""Within-fold inventory cost contrasts for two case-pack sizes."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from figure_style import format_figure  # noqa: E402


def main():
    data = pd.read_csv(ROOT / "m5/results/inventory_replay/case_pack_by_fold.csv")
    costs = data.assign(cents=100 * data.cost / data.weeks_items)
    paired = costs.pivot(index=["lead", "forecast", "fold"], columns="case_pack", values="cents")
    difference = paired[[6, 12]].sub(paired[1], axis=0)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.8, 5.2), layout="constrained")
    for row, (model, label) in enumerate([("naive", "上周销量"), ("q50", "中位数预测")]):
        for col, lead in enumerate((1, 2)):
            ax = axes[row, col]
            pairs = difference.loc[(lead, model), :].sort_index()
            ax.axhline(0, color="#565656", ls=":", lw=.85)
            for values in pairs.to_numpy():
                ax.plot([6, 12], values, color="#91999e", lw=.85, marker="o", ms=3,
                        markerfacecolor="white", alpha=.9)
            ax.plot([6, 12], np.median(pairs, axis=0), color="#224c66", lw=1.55,
                    marker="D", ms=4.5, label="各箱规的六折中位数")
            ax.set(xlim=(5,13), xticks=[6,12], yticks=[-20,-15,-10,-5,0])
            ax.tick_params(labelbottom=True)
            ax.set_title(f"{'ABCD'[row*2+col]}    {label} · 提前期 {lead} 周", loc="left", fontsize=11)
            ax.spines[["top","right"]].set_visible(False)
    fig.supxlabel("每箱件数", fontsize=11)
    fig.supylabel("相对逐件补货的费用差（美分/商品门店周）", fontsize=11)
    format_figure(fig)
    fig.savefig(ROOT / "images/case_pack.png", dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    main()
