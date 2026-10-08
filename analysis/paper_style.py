"""Shared look of the paper's figures (paper/figures/*.pdf): STIX like the paper's text, sized for its 15.5 cm text block."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXT_WIDTH = 15.5 / 2.54  # inches, the template's text block
SMALL = 8.5  # annotations and values; axis labels and legends are 9 pt, the \footnotesize of the 11 pt text
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"  # the paper's page is white
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3, validated all-pairs on white
NAMES = {"mhc-h1": "mHC", "ctrl-mlp1216": "MLP control", "ctrl-mlp2240": "MLP control", "residual": "residual",
         "a-global-h4": "global $h=4$", "a-local-h4": "local $h=4$"}  # run names -> the paper's names


def use() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "pdf.fonttype": 42,
    })
