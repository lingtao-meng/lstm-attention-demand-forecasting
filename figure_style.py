"""Shared typography and axis treatment for the project's research figures."""

from functools import lru_cache


@lru_cache(maxsize=1)
def paper_fonts():
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    families = ("Noto Serif CJK SC", "Songti SC", "SimSun", "Noto Sans CJK SC", "Microsoft YaHei", "STHeiti")
    return ["STIXGeneral", *[name for name in families if name in installed][:1]]


def format_figure(fig):
    """Format existing artists without changing coordinates, bins, limits or legend order."""
    from matplotlib.text import Text

    fig.set_facecolor("white")
    for axis in fig.axes:
        axis.set_facecolor("white")
        if axis.name != "rectilinear" or getattr(axis, "_colorbar", None) is not None:
            continue
        axis.tick_params(direction="in", length=3.2, width=.65, colors="#252525")
        for spine in axis.spines.values():
            spine.set_linewidth(.65)
            spine.set_color("#454545")
        for line in [*axis.get_xgridlines(), *axis.get_ygridlines()]:
            line.set_color("#dddddd")
            line.set_linewidth(.45)
        legend = axis.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
        axis.title.set_fontweight("normal")
    for text in fig.findobj(Text):
        text.set_fontfamily(paper_fonts())
    return fig


def format_plotly(fig):
    """Set publication colours and axes while preserving hover and secondary scales."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Georgia, Songti SC, Noto Serif CJK SC, serif", size=13, color="#262b2e"),
        colorway=["#355d78", "#976b49", "#66715c", "#776585", "#778c99"],
        paper_bgcolor="white", plot_bgcolor="white",
    )
    axes = dict(ticks="inside", ticklen=4, showline=True, linewidth=1,
                linecolor="#657078", gridcolor="#e5e5e5", zerolinecolor="#9aa0a3")
    fig.update_xaxes(**axes)
    fig.update_yaxes(**axes)
    return fig
