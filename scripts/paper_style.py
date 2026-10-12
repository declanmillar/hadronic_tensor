"""Shared matplotlib style for all paper figures: Computer Modern (LaTeX)
fonts throughout -- via matplotlib's bundled CM fonts and mathtext, NOT
text.usetex (the usetex/dviread path on this box silently drops symbol-font
glyphs: minus signs, arrows, angle brackets) -- plus the Okabe-Ito palette
and a uniform verbatim-font device-name helper.

Usage in a figure script (replaces any local rcParams block):

    from paper_style import OI, PALE, MARKERS, dev
    ...
    ax.set_title(dev("kingston"))
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],               # bundled Computer Modern text font
    "mathtext.fontset": "cm",              # Computer Modern math
    "axes.formatter.use_mathtext": True,   # tick labels in CM
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

# Okabe-Ito, in the paper's order
OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7",
      "#000000"]
# Signed tensor heatmaps: vermillion (negative), white, blue (positive).
OI_DIV = LinearSegmentedColormap.from_list(
    "oi_div", [OI[1], "#FFFFFF", OI[0]])
PALE = ["#80B9D9", "#EAAF80"]          # 50%-white OI[0], OI[1]
# marker cycle for per-volume series (B/W-distinguishable)
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


def dev(name: str) -> str:
    """Device name in verbatim font: dev('kingston') -> ibm_kingston in tt."""
    return r"$\mathtt{ibm\_" + name + "}$"
