"""高级分析图集：指纹热图、组成评价、埋藏面积、几何统计、接触矩阵、综合仪表盘。

所有图中文字为英文；报告中的中英文解释由 report.py 生成。
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg", force=False)
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import MaxNLocator

from .analysis import PLResult, PPResult
from .descriptors import (
    CLASS_CN, CLASS_EN, CLASS_ORDER, PLAssessment, PPAssessment,
    composition, ligand_descriptors, residue_class,
)
from .render import COLORS, KIND_LABEL_EN
from .sasa import BindingSASA

FONT = "DejaVu Sans"
BOOST = 1.40
INK = "#1F2933"
MUTED = "#5B6570"
GRID = "#DDE4E0"
SOFT = "#F7FAF8"
BORDER = "#DCE6DF"
PRIMARY = "#174534"
TYPE_COLORS = {
    "hbond": "#2E7D32",
    "salt": "#7B1FA2",
    "pi": "#00796B",
    "metal": "#6D4C41",
    "hydrophobic": "#AE3B2C",
}
TYPE_LABELS = {
    "hbond": "Hydrogen bond",
    "salt": "Salt bridge",
    "pi": "π–π stacking",
    "metal": "Metal coordination",
    "hydrophobic": "Hydrophobic contact",
}
TYPE_ORDER = ["hbond", "salt", "pi", "metal", "hydrophobic"]
CLASS_COLORS = {
    "Hydrophobic": "#C0604A",
    "Aromatic": "#B8892D",
    "Polar": "#3F7A8C",
    "Positive": "#6A5ACD",
    "Negative": "#B03A5B",
    "Glycine": "#8A9A90",
}
CHAIN_A = "#3D6D96"
CHAIN_B = "#3F7A4F"


def _figure(width: float, height: float, scale: float, dpi: float) -> Figure:
    fig = Figure(figsize=(width * scale, height * scale), dpi=dpi)
    fig.patch.set_facecolor("white")
    return fig


def _titles(fig, title: str, subtitle: str, scale: float):
    fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=12.4 * scale * BOOST,
             fontweight="bold", color=INK, family=FONT)
    fig.text(0.5, 0.916, subtitle, ha="center", va="top", fontsize=7.4 * scale * BOOST,
             color=MUTED, family=FONT)


def _panel(ax, title: str, scale: float):
    ax.set_title(title, loc="left", fontsize=9.4 * scale * BOOST, fontweight="bold",
                 color=INK, family=FONT, pad=8)


def _clean_axes(ax, scale: float):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7.2 * scale * BOOST, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6 * scale)
    ax.set_axisbelow(True)


def _type_counts(stats: list[dict]) -> dict[str, int]:
    return {
        k: sum(int(s.get(k, 0)) for s in stats) for k in TYPE_ORDER
    }


def _dominant(counts: dict[str, int]) -> str:
    present = [(v, k) for k, v in counts.items() if v > 0]
    if not present:
        return "hbond"
    return max(present)[1]


def _short_label(label: str) -> str:
    return label.replace(" (", "(").replace(")", ")")


def _residue_rows(result, limit: int = 18) -> list[dict]:
    stats = [s for s in result.residue_stats if s["total"] > 0]
    stats = stats[:limit]
    return stats


def build_fingerprint_figure(result: PLResult | PPResult, scale: float = 1.0,
                             dpi: float = 110) -> Figure:
    stats = [s for s in result.residue_stats if s["total"] > 0]
    if not stats:
        fig = _figure(7.2, 3.2, scale, dpi)
        fig.text(0.5, 0.5, "No residue-level interactions", ha="center",
                 color=MUTED, family=FONT, fontsize=10 * scale * BOOST)
        return fig
    max_rows = 12
    if len(stats) > max_rows:
        head = stats[:max_rows]
        tail = stats[max_rows:]
        merged = {"label": f"Other ({len(tail)})"}
        for key in TYPE_ORDER:
            merged[key] = sum(s[key] for s in tail)
        merged["total"] = sum(s["total"] for s in tail)
        merged["min_dist"] = min(s["min_dist"] for s in tail)
        stats = head + [merged]

    matrix = np.array([[s[k] for k in TYPE_ORDER] for s in stats], dtype=float)
    totals = matrix.sum(axis=1)
    n = len(stats)
    height = max(4.8, 0.28 * n + 1.9)
    fig = _figure(9.6, height, scale, dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.8, 1.05], wspace=0.06,
                          left=0.26, right=0.965, top=0.82, bottom=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "ligcn", ["#FFFFFF", "#E7F1EA", "#8FBF9F", "#2E7D5A", "#174534"]
    )
    im = ax1.imshow(matrix, aspect="auto", cmap=cmap, vmin=0,
                    vmax=max(1.0, matrix.max()))
    for i in range(n):
        for j in range(len(TYPE_ORDER)):
            value = int(matrix[i, j])
            if value == 0:
                ax1.text(j, i, "·", ha="center", va="center",
                         color="#A6B2AA", fontsize=7.0 * scale * BOOST, family=FONT)
            else:
                color = "white" if matrix[i, j] > matrix.max() * 0.55 else INK
                ax1.text(j, i, str(value), ha="center", va="center",
                         color=color, fontsize=7.4 * scale * BOOST, fontweight="bold",
                         family=FONT)
    ax1.set_xticks(np.arange(len(TYPE_ORDER)))
    ax1.set_xticklabels([TYPE_LABELS[k].replace(" ", "\n", 1) for k in TYPE_ORDER],
                        fontsize=6.8 * scale * BOOST, family=FONT)
    ax1.set_yticks(np.arange(n))
    ax1.set_yticklabels([s["label"] for s in stats], fontsize=7.3 * scale * BOOST,
                        family=FONT)
    ax1.set_xticks(np.arange(-0.5, len(TYPE_ORDER), 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax1.grid(which="minor", color="white", linewidth=1.2 * scale)
    ax1.tick_params(which="minor", length=0)
    ax1.tick_params(axis="y", length=0, pad=4)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    _panel(ax1, "Residue × interaction-type fingerprint", scale)

    colors = [TYPE_COLORS[_dominant({k: s[k] for k in TYPE_ORDER})] for s in stats]
    ax2.barh(np.arange(n), totals, color=colors, height=0.62, edgecolor="white",
             linewidth=0.5 * scale)
    for i, total in enumerate(totals):
        ax2.text(total + max(totals) * 0.03, i, f"{int(total)}",
                 va="center", ha="left", fontsize=6.8 * scale * BOOST,
                 color=INK, family=FONT)
    ax2.set_xlim(0, max(totals) * 1.22 if len(totals) else 1)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel("Total", fontsize=7.4 * scale * BOOST, color=MUTED, family=FONT)
    ax2.tick_params(axis="y", left=False, labelleft=False)
    _clean_axes(ax2, scale)
    _panel(ax2, "Total records", scale)

    _titles(fig, "Interaction Fingerprint",
            f"PDB {result.structure.pdb_id}   ·   {len(stats)} residues   ·   "
            f"{len(result.interactions)} interaction records", scale)
    fig.text(0.985, 0.015,
             "Each cell is the number of records; · denotes zero. Top 12 residues shown.",
             ha="right", va="bottom", fontsize=6.2 * scale * BOOST, color="#9AA4AE",
             family=FONT)
    return fig


def build_composition_figure(result: PLResult | PPResult,
                             assessment: PLAssessment | PPAssessment,
                             scale: float = 1.0, dpi: float = 110) -> Figure:
    fig = _figure(10.2, 5.1, scale, dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.38,
                          left=0.15, right=0.965, top=0.81, bottom=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    if isinstance(result, PLResult):
        comp = assessment.pocket_composition
        background = assessment.background_composition
        label_front = "Pocket"
        bg_label = "Protein"
        subtitle = (f"Pocket residues {comp.total}   ·   protein background "
                    f"{background.total}   ·   ligand {result.ligand.resname}")
    else:
        comp = assessment.composition_all
        background = assessment.background_composition
        label_front = "Interface"
        bg_label = "Protein"
        subtitle = (f"Interface residues {comp.total}   ·   background "
                    f"{background.total}   ·   chains {result.chain_a.strip()}/"
                    f"{result.chain_b.strip()}")

    y = np.arange(len(CLASS_ORDER))
    front_vals = [comp.percentages[c] for c in CLASS_ORDER]
    bg_vals = [background.percentages[c] for c in CLASS_ORDER]
    ax1.barh(y + 0.19, front_vals, height=0.34,
             color=[CLASS_COLORS[c] for c in CLASS_ORDER], edgecolor="white",
             linewidth=0.6 * scale, label=label_front)
    ax1.barh(y - 0.19, bg_vals, height=0.34, color="#D5DED8",
             edgecolor="white", linewidth=0.6 * scale, label=bg_label)
    for yi, value in zip(y + 0.19, front_vals):
        if value > 0:
            ax1.text(value + 1.0, yi, f"{value:.0f}%", va="center",
                     fontsize=6.4 * scale * BOOST, color=INK, family=FONT)
    ax1.set_yticks(y)
    ax1.set_yticklabels([CLASS_EN[c] for c in CLASS_ORDER],
                        fontsize=7.1 * scale * BOOST, family=FONT)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(60, max(front_vals + bg_vals) * 1.25))
    ax1.set_xlabel("Residue proportion (%)", fontsize=7.6 * scale * BOOST,
                   color=MUTED, family=FONT)
    ax1.legend(frameon=False, fontsize=7.0 * scale * BOOST, loc="lower right")
    _clean_axes(ax1, scale)
    _panel(ax1, "Residue class composition", scale)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.add_patch(Rectangle((0, 0), 1, 1, transform=ax2.transAxes,
                            facecolor=SOFT, edgecolor=BORDER,
                            linewidth=0.7 * scale))
    _panel(ax2, "Binding profile", scale)
    if isinstance(result, PLResult):
        d = assessment.descriptors
        rows = [
            ("Formula", d.formula or "—"),
            ("Mass (observed)", f"{d.mass:.1f} Da"),
            ("Heavy atoms", f"{d.heavy_atoms}"),
            ("Rings (aromatic)", f"{d.rings} ({d.aromatic_rings})"),
            ("Rotatable bonds", f"{d.rotatable_bonds}"),
            ("H-bond donor / acceptor", f"{d.donors} / {d.acceptors}"),
            ("Formal charge", f"{d.formal_charge:+d}"),
            ("BSA (half-sum)", f"{assessment.bsa:.1f} Å²"),
        ]
        y0 = 0.88
        for i, (key, value) in enumerate(rows):
            yy = y0 - i * 0.075
            ax2.text(0.06, yy, key, transform=ax2.transAxes, fontsize=7.0 * scale * BOOST,
                     color=MUTED, family=FONT, va="center")
            ax2.text(0.94, yy, value, transform=ax2.transAxes, fontsize=7.4 * scale * BOOST,
                     color=INK, family=FONT, va="center", ha="right")
        frac = max(0.0, min(1.0, d.burial_fraction))
        ax2.add_patch(Rectangle((0.06, 0.12), 0.88, 0.055,
                                transform=ax2.transAxes, facecolor="#E3EAE5",
                                edgecolor="none"))
        ax2.add_patch(Rectangle((0.06, 0.12), 0.88 * frac, 0.055,
                                transform=ax2.transAxes, facecolor="#2E7D5A",
                                edgecolor="none"))
        ax2.text(0.06, 0.20, "Ligand burial", transform=ax2.transAxes,
                 fontsize=7.0 * scale * BOOST, color=MUTED, family=FONT)
        ax2.text(0.94, 0.20, f"{100 * frac:.1f}%", transform=ax2.transAxes,
                 fontsize=7.6 * scale * BOOST, color=INK, family=FONT, ha="right")
        ax2.text(0.06, 0.055,
                 f"Buried {d.buried_area:.1f} Å² of free SASA {d.sasa_free:.1f} Å²",
                 transform=ax2.transAxes, fontsize=6.5 * scale * BOOST,
                 color=MUTED, family=FONT)
    else:
        side_a = assessment.sasa.buried_a
        side_b = assessment.sasa.buried_b
        values = [side_a, side_b]
        labels = [f"Chain {result.chain_a.strip()}", f"Chain {result.chain_b.strip()}"]
        colors = [CHAIN_A, CHAIN_B]
        maxv = max(values + [1.0])
        for label, value, color, yy in zip(labels, values, colors, (0.70, 0.53)):
            ax2.text(0.06, yy + 0.075, label, transform=ax2.transAxes,
                     fontsize=7.2 * scale * BOOST, color=MUTED, family=FONT)
            ax2.add_patch(Rectangle((0.06, yy), 0.62 * (value / maxv), 0.05,
                                    transform=ax2.transAxes, facecolor=color,
                                    edgecolor="none"))
            ax2.text(0.70, yy + 0.025, f"{value:.0f} Å²", transform=ax2.transAxes,
                     fontsize=7.0 * scale * BOOST, color=INK, family=FONT, va="center",
                     ha="left")
        class_text = {"small": "Small", "medium": "Medium", "large": "Large"}
        ax2.text(0.06, 0.40,
                 f"Total buried  {assessment.total_buried:.0f} Å²",
                 transform=ax2.transAxes, fontsize=8.0 * scale * BOOST,
                 color=INK, family=FONT, fontweight="bold")
        ax2.text(0.06, 0.33,
                 f"BSA (half-sum)  {assessment.bsa:.0f} Å²",
                 transform=ax2.transAxes, fontsize=7.2 * scale * BOOST,
                 color=MUTED, family=FONT)
        ax2.text(0.06, 0.26,
                 f"Interface size  {class_text.get(assessment.interface_size_class, '—')}",
                 transform=ax2.transAxes, fontsize=7.2 * scale * BOOST,
                 color=INK, family=FONT)
        ax2.text(0.06, 0.18,
                 f"Interface residues  {len(result.interface_a)} + "
                 f"{len(result.interface_b)}",
                 transform=ax2.transAxes, fontsize=7.0 * scale * BOOST,
                 color=MUTED, family=FONT)
        ax2.text(0.06, 0.08,
                 "Class composition is shown in the left panel.",
                 transform=ax2.transAxes, fontsize=6.6 * scale * BOOST,
                 color="#8C9A92", family=FONT)

    _titles(fig, "Binding Interface Composition and Properties", subtitle, scale)
    fig.text(0.985, 0.015,
             "Residue classes are mutually exclusive; percentages are within each set.",
             ha="right", va="bottom", fontsize=6.2 * scale * BOOST, color="#9AA4AE",
             family=FONT)
    return fig


def _residue_labels(structure) -> dict[tuple, str]:
    return {r.key: r.label_spaced for r in structure.residues}


def build_bsa_figure(result: PLResult | PPResult,
                     assessment: PLAssessment | PPAssessment,
                     scale: float = 1.0, dpi: float = 110) -> Figure:
    labels = _residue_labels(result.structure)
    fig = _figure(10.2, 5.3, scale, dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.48,
                          left=0.23, right=0.965, top=0.81, bottom=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    if isinstance(result, PLResult):
        items = [(labels.get(k, str(k)), v) for k, v in
                 assessment.pocket_residue_bsa.items() if v > 0.2]
        items.sort(key=lambda x: -x[1])
        items = items[:12]
        colors = ["#C0604A"] * len(items)
        subtitle = (f"PDB {result.structure.pdb_id}   ·   ligand "
                    f"{result.ligand.resname}   ·   "
                    f"total buried {assessment.total_buried:.0f} Å²")
    else:
        items = []
        for key, value in assessment.residue_bsa_a.items():
            if value > 0.2:
                items.append((labels.get(key, str(key)), value, "A"))
        for key, value in assessment.residue_bsa_b.items():
            if value > 0.2:
                items.append((labels.get(key, str(key)), value, "B"))
        items.sort(key=lambda x: -x[1])
        items = items[:12]
        colors = [CHAIN_A if x[2] == "A" else CHAIN_B for x in items]
        subtitle = (f"PDB {result.structure.pdb_id}   ·   chains "
                    f"{result.chain_a.strip()}/{result.chain_b.strip()}   ·   "
                    f"total buried {assessment.total_buried:.0f} Å²")

    if not items:
        ax1.axis("off")
        ax1.text(0.5, 0.5, "No significant per-residue burial", ha="center",
                 va="center", color=MUTED, family=FONT, fontsize=9 * scale * BOOST)
        _panel(ax1, "Per-residue burial", scale)
        _titles(fig, "Buried Surface Area (BSA) Analysis", subtitle, scale)
        return fig

    y = np.arange(len(items))[::-1]
    values = [x[1] for x in items]
    ax1.barh(y, values, color=colors, height=0.62, edgecolor="white",
             linewidth=0.5 * scale)
    for yi, value in zip(y, values):
        ax1.text(value + max(values) * 0.01, yi, f"{value:.0f}",
                 va="center", ha="left", fontsize=6.3 * scale * BOOST,
                 color=INK, family=FONT)
    ax1.set_yticks(y)
    ax1.set_yticklabels([x[0] for x in items], fontsize=7.0 * scale * BOOST,
                        family=FONT)
    ax1.set_xlim(0, max(values) * 1.25 if values else 1)
    ax1.set_xlabel("Buried surface area per residue (Å²)", fontsize=7.6 * scale * BOOST,
                   color=MUTED, family=FONT)
    _clean_axes(ax1, scale)
    _panel(ax1, "Per-residue burial", scale)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.add_patch(Rectangle((0, 0), 1, 1, transform=ax2.transAxes,
                            facecolor=SOFT, edgecolor=BORDER,
                            linewidth=0.7 * scale))
    _panel(ax2, "Surface area balance", scale)
    if isinstance(result, PLResult):
        d = assessment.descriptors
        ax2.text(0.06, 0.82, "Binding buried surface", transform=ax2.transAxes,
                 fontsize=7.2 * scale * BOOST, color=MUTED, family=FONT)
        ax2.text(0.06, 0.72, f"{assessment.bsa:.1f} Å²", transform=ax2.transAxes,
                 fontsize=13.0 * scale * BOOST, color="#1F5B45", fontweight="bold",
                 family=FONT)
        ax2.text(0.06, 0.65, f"Total buried {assessment.total_buried:.1f} Å²",
                 transform=ax2.transAxes, fontsize=6.6 * scale * BOOST, color=MUTED,
                 family=FONT)
        parts = [("Protein buried", assessment.sasa.buried_a, "#3D6D96"),
                 ("Ligand buried", assessment.sasa.buried_b, "#C0604A")]
        ypos = [0.48, 0.36]
        maxv = max(assessment.sasa.buried_a, assessment.sasa.buried_b)
        for (label, value, color), yy in zip(parts, ypos):
            ax2.text(0.06, yy + 0.075, label, transform=ax2.transAxes,
                     fontsize=6.7 * scale * BOOST, color=MUTED, family=FONT)
            ax2.add_patch(Rectangle((0.06, yy), 0.62 * (value / maxv), 0.045,
                                    transform=ax2.transAxes, facecolor=color,
                                    edgecolor="none"))
            ax2.text(0.70, yy + 0.022, f"{value:.0f} Å²", transform=ax2.transAxes,
                     fontsize=6.7 * scale * BOOST, color=INK, family=FONT, ha="left",
                     va="center")
        frac = max(0.0, min(1.0, d.burial_fraction))
        ax2.add_patch(Rectangle((0.06, 0.13), 0.86, 0.045,
                                transform=ax2.transAxes, facecolor="#E4EBE6",
                                edgecolor="none"))
        ax2.add_patch(Rectangle((0.06, 0.13), 0.86 * frac, 0.045,
                                transform=ax2.transAxes, facecolor="#2E7D5A",
                                edgecolor="none"))
        ax2.text(0.06, 0.19, "Ligand burial fraction", transform=ax2.transAxes,
                 fontsize=6.7 * scale * BOOST, color=MUTED, family=FONT)
        ax2.text(0.94, 0.19, f"{100 * frac:.0f}%", transform=ax2.transAxes,
                 fontsize=7.0 * scale * BOOST, color=INK, family=FONT, ha="right")
        ax2.text(0.06, 0.055,
                 f"Free {d.sasa_free:.1f} Å²  ·  bound {d.sasa_bound:.1f} Å²",
                 transform=ax2.transAxes, fontsize=6.4 * scale * BOOST, color=MUTED,
                 family=FONT)
    else:
        a = assessment.sasa.buried_a
        b = assessment.sasa.buried_b
        ax2.text(0.06, 0.82, "Interface area (half-sum)", transform=ax2.transAxes,
                 fontsize=7.0 * scale * BOOST, color=MUTED, family=FONT)
        ax2.text(0.06, 0.72, f"{assessment.bsa:.1f} Å²", transform=ax2.transAxes,
                 fontsize=13.0 * scale * BOOST, color="#1F5B45", fontweight="bold",
                 family=FONT)
        ax2.text(0.06, 0.65, f"Total buried {assessment.total_buried:.1f} Å²",
                 transform=ax2.transAxes, fontsize=6.6 * scale * BOOST, color=MUTED,
                 family=FONT)
        bars = [(f"Chain {result.chain_a.strip()}", a, CHAIN_A),
                (f"Chain {result.chain_b.strip()}", b, CHAIN_B)]
        ypos = [0.47, 0.35]
        maxv = max(a, b)
        for (label, value, color), yy in zip(bars, ypos):
            ax2.text(0.06, yy + 0.075, label, transform=ax2.transAxes,
                     fontsize=6.8 * scale * BOOST, color=MUTED, family=FONT)
            ax2.add_patch(Rectangle((0.06, yy), 0.62 * (value / maxv), 0.045,
                                    transform=ax2.transAxes, facecolor=color,
                                    edgecolor="none"))
            ax2.text(0.70, yy + 0.022, f"{value:.0f} Å²", transform=ax2.transAxes,
                     fontsize=6.7 * scale * BOOST, color=INK, family=FONT, ha="left",
                     va="center")
        classes = {"small": "Small (<600 Å² total)",
                   "medium": "Medium (600–1200 Å²)",
                   "large": "Large (>1200 Å²)"}
        ax2.text(0.06, 0.22, "Interface size class", transform=ax2.transAxes,
                 fontsize=6.7 * scale * BOOST, color=MUTED, family=FONT)
        ax2.text(0.06, 0.14,
                 classes.get(assessment.interface_size_class, "—"),
                 transform=ax2.transAxes, fontsize=7.2 * scale * BOOST, color=INK,
                 family=FONT)
        ax2.text(0.06, 0.06,
                 f"Interface residues {len(result.interface_a)} + {len(result.interface_b)}",
                 transform=ax2.transAxes, fontsize=6.6 * scale * BOOST, color=MUTED,
                 family=FONT)

    _titles(fig, "Buried Surface Area (BSA) Analysis", subtitle, scale)
    fig.text(0.985, 0.015,
             "Shrake–Rupley SASA, probe 1.4 Å. Half-sum BSA is the conventional two-sided interface area.",
             ha="right", va="bottom", fontsize=6.2 * scale * BOOST, color="#9AA4AE",
             family=FONT)
    return fig


def build_geometry_figure(result: PLResult | PPResult, scale: float = 1.0,
                          dpi: float = 110) -> Figure:
    hbonds = [it for it in result.interactions if it.kind == "hbond"]
    hydrophobic = [it for it in result.interactions if it.kind == "hydrophobic"]
    fig = _figure(10.2, 5.0, scale, dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.40,
                          left=0.11, right=0.965, top=0.81, bottom=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    if hbonds:
        d = [it.distance for it in hbonds]
        ang = [it.angle if it.angle is not None else 0 for it in hbonds]
        ax1.scatter(d, ang, s=26 * scale ** 2, color=TYPE_COLORS["hbond"],
                    edgecolor="white", linewidth=0.6 * scale, alpha=0.9, zorder=4)
        ax1.axvline(3.5, color="#B7C4BB", linewidth=0.8 * scale,
                    linestyle=(0, (2, 2)), zorder=1)
        ax1.axhline(120, color="#B7C4BB", linewidth=0.8 * scale,
                    linestyle=(0, (2, 2)), zorder=1)
        ax1.text(3.53, 176, "3.5 Å", fontsize=6.6 * scale * BOOST, color=MUTED,
                 family=FONT, va="top")
        ax1.text(min(d) + 0.03, 123, "120°", fontsize=6.4 * scale * BOOST,
                 color=MUTED, family=FONT)
        med_d = float(np.median(d))
        med_a = float(np.median(ang))
        ax1.text(0.97, 0.06,
                 f"n = {len(hbonds)}   median {med_d:.2f} Å / {med_a:.0f}°",
                 transform=ax1.transAxes, ha="right", fontsize=7.0 * scale * BOOST,
                 color=MUTED, family=FONT)
        ax1.set_xlim(max(2.4, min(d) - 0.15), max(4.2, max(d) + 0.15))
        ax1.set_ylim(70, 180)
        ax1.set_yticks([80, 100, 120, 140, 160])
        ax1.set_xlabel("Donor–acceptor distance (Å)", fontsize=7.6 * scale * BOOST,
                       color=MUTED, family=FONT)
        ax1.set_ylabel("D–H···A angle (°)", fontsize=7.6 * scale * BOOST,
                       color=MUTED, family=FONT)
        ax1.yaxis.grid(True, color=GRID, linewidth=0.6 * scale)
        _clean_axes(ax1, scale)
    else:
        ax1.axis("off")
        ax1.text(0.5, 0.5, "No hydrogen bonds detected", ha="center",
                 color=MUTED, family=FONT, fontsize=9 * scale * BOOST)
    _panel(ax1, "Hydrogen-bond geometry", scale)

    if hydrophobic:
        per_res = {}
        for it in hydrophobic:
            key = it.res_a.label_spaced
            entry = per_res.setdefault(key, {"count": 0, "min": it.distance})
            entry["count"] += it.contacts
            entry["min"] = min(entry["min"], it.distance)
        items = sorted(per_res.items(), key=lambda kv: -kv[1]["count"])[:12]
        labels = [k for k, _ in items]
        counts = [v["count"] for _, v in items]
        mins = [v["min"] for _, v in items]
        y = np.arange(len(items))[::-1]
        norm = (np.array(mins) - min(mins)) / max(max(mins) - min(mins), 1e-6)
        colors = [
            matplotlib.colors.to_hex(
                matplotlib.colors.LinearSegmentedColormap.from_list(
                    "hyd", ["#E8B4A8", "#AE3B2C"])(float(t))
            ) for t in norm
        ]
        ax2.barh(y, counts, color=colors, height=0.6, edgecolor="white",
                 linewidth=0.5 * scale)
        for yi, count, md in zip(y, counts, mins):
            ax2.text(count + max(counts) * 0.02, yi, f"{count}  ({md:.1f} Å)",
                     va="center", ha="left", fontsize=6.2 * scale * BOOST,
                     color=INK, family=FONT)
        ax2.set_yticks(y)
        ax2.set_yticklabels(labels, fontsize=6.9 * scale * BOOST, family=FONT)
        ax2.set_xlim(0, max(counts) * 1.35)
        ax2.set_xlabel("Hydrophobic atom-pair contacts", fontsize=7.6 * scale * BOOST,
                       color=MUTED, family=FONT)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        _clean_axes(ax2, scale)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "No hydrophobic contacts detected", ha="center",
                 color=MUTED, family=FONT, fontsize=9 * scale * BOOST)
    _panel(ax2, "Hydrophobic contact multiplicity", scale)

    _titles(fig, "Interaction Geometry and Contact Statistics",
            f"PDB {result.structure.pdb_id}   ·   "
            f"{len(hbonds)} H-bonds   ·   {len(hydrophobic)} hydrophobic residue contacts",
            scale)
    fig.text(0.985, 0.015,
             "Hydrophobic bar labels show atom-pair contacts and the minimum distance per residue.",
             ha="right", va="bottom", fontsize=6.2 * scale * BOOST, color="#9AA4AE",
             family=FONT)
    return fig


def build_contact_figure(result: PLResult | PPResult,
                         assessment: PLAssessment | PPAssessment,
                         scale: float = 1.0, dpi: float = 110) -> Figure:
    if isinstance(result, PLResult):
        return _ligand_burial_figure(result, assessment, scale, dpi)
    return _contact_matrix_figure(result, scale, dpi)


def _ligand_burial_figure(result: PLResult, assessment: PLAssessment,
                          scale: float, dpi: float) -> Figure:
    ligand = result.ligand
    atoms = ligand.heavy_atoms
    buried = assessment.sasa.per_atom_buried_b
    free_per_atom = buried + assessment.sasa.per_atom_complex_b
    fractions = np.divide(buried, free_per_atom,
                          out=np.zeros_like(buried), where=free_per_atom > 1e-6)
    contact_atoms = set()
    for it in result.interactions:
        if it.atom_b and it.atom_b != "ring":
            contact_atoms.add(it.atom_b)
    items = []
    for i, atom in enumerate(atoms):
        if buried[i] <= 0.05:
            continue
        items.append((atom.name, float(buried[i]), float(fractions[i]),
                      atom.element.upper(), atom.name in contact_atoms))
    items.sort(key=lambda x: -x[1])
    items = items[:12]
    if not items:
        fig = _figure(7.2, 3.2, scale, dpi)
        fig.text(0.5, 0.5, "No ligand atom burial detected", ha="center",
                 va="center", color=MUTED, family=FONT, fontsize=10 * scale * BOOST)
        return fig

    fig = _figure(10.1, 5.1, scale, dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.42,
                          left=0.14, right=0.965, top=0.81, bottom=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    y = np.arange(len(items))[::-1]
    colors = []
    for _, _, _, element, _ in items:
        colors.append({
            "C": "#4A5652", "N": "#2C5AA0", "O": "#D93B30",
            "S": "#C99A00", "P": "#E07B24",
        }.get(element, "#7A7A7A"))
    values = [x[1] for x in items]
    ax1.barh(y, values, color=colors, height=0.62, edgecolor="white",
             linewidth=0.5 * scale)
    for yi, item in zip(y, items):
        name, value, frac, element, is_contact = item
        marker = " ●" if is_contact else ""
        ax1.text(value + max(values) * 0.015, yi,
                 f"{value:.0f} Å²  ({100 * frac:.0f}%){marker}",
                 va="center", ha="left", fontsize=6.1 * scale * BOOST,
                 color=INK, family=FONT)
    ax1.set_yticks(y)
    ax1.set_yticklabels([x[0] for x in items], fontsize=7.0 * scale * BOOST, family=FONT)
    ax1.set_xlim(0, max(values) * 1.55 if values else 1)
    ax1.set_xlabel("Buried SASA per ligand atom (Å²)", fontsize=7.6 * scale * BOOST,
                   color=MUTED, family=FONT)
    _clean_axes(ax1, scale)
    _panel(ax1, "Ligand atom burial profile", scale)

    elements = {}
    for atom in atoms:
        elements[atom.element.upper()] = elements.get(atom.element.upper(), 0) + 1
    order = [el for el in ("C", "N", "O", "S", "P", "F", "CL", "BR", "I")
             if el in elements]
    values = [elements[el] for el in order]
    colors = [{
        "C": "#4A5652", "N": "#2C5AA0", "O": "#D93B30", "S": "#C99A00",
        "P": "#E07B24", "F": "#2FA84F", "CL": "#2FA84F", "BR": "#8C4A2F",
        "I": "#6B3FA0",
    }.get(el, "#7A7A7A") for el in order]
    wedges, _ = ax2.pie(values, colors=colors, startangle=90,
                        counterclock=False,
                        wedgeprops=dict(width=0.34, edgecolor="white",
                                        linewidth=1.2 * scale))
    ax2.text(0, 0, f"{len(atoms)}\nheavy atoms", ha="center", va="center",
             fontsize=8.0 * scale * BOOST, color=INK, family=FONT)
    ax2.legend(wedges, [f"{el}  {v}" for el, v in zip(order, values)],
               loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
               frameon=False, fontsize=6.2 * scale * BOOST)
    _panel(ax2, "Ligand element composition", scale)

    _titles(fig, "Ligand Burial and Composition",
            f"PDB {result.structure.pdb_id}   ·   ligand {ligand.resname}   ·   "
            f"● marks atoms in polar interactions", scale)
    fig.text(0.985, 0.015,
             "Burial fraction = buried SASA / free-atom SASA; computed with a 1.4 Å probe.",
             ha="right", va="bottom", fontsize=6.2 * scale * BOOST, color="#9AA4AE",
             family=FONT)
    return fig


def _contact_matrix_figure(result: PPResult, scale: float, dpi: float) -> Figure:
    groups: dict[tuple, dict] = {}
    for it in result.interactions:
        if it.res_b is None:
            continue
        key = (it.res_a.key, it.res_b.key)
        entry = groups.setdefault(key, {"count": 0, "kinds": {}})
        entry["count"] += 1
        entry["kinds"][it.kind] = entry["kinds"].get(it.kind, 0) + 1
    if not groups:
        fig = _figure(7.2, 3.2, scale, dpi)
        fig.text(0.5, 0.5, "No residue–residue interactions", ha="center",
                 color=MUTED, family=FONT, fontsize=10 * scale * BOOST)
        return fig
    total_a = {}
    total_b = {}
    for (ka, kb), entry in groups.items():
        total_a[ka] = total_a.get(ka, 0) + entry["count"]
        total_b[kb] = total_b.get(kb, 0) + entry["count"]
    keys_a = sorted(total_a, key=lambda k: -total_a[k])[:12]
    keys_b = sorted(total_b, key=lambda k: -total_b[k])[:12]
    idx_a = {k: i for i, k in enumerate(keys_a)}
    idx_b = {k: i for i, k in enumerate(keys_b)}
    label_map = _residue_labels(result.structure)
    points = []
    for (ka, kb), entry in groups.items():
        if ka in idx_a and kb in idx_b:
            kind = _dominant({k: entry["kinds"].get(k, 0) for k in TYPE_ORDER})
            points.append((idx_a[ka], idx_b[kb], entry["count"], kind))
    fig = _figure(10.2, 5.6, scale, dpi)
    ax = fig.add_axes([0.19, 0.22, 0.56, 0.60])
    for i in range(len(keys_a)):
        ax.axhline(i, color="#EDF2EE", linewidth=0.7 * scale, zorder=0)
    for j in range(len(keys_b)):
        ax.axvline(j, color="#EDF2EE", linewidth=0.7 * scale, zorder=0)
    counts = [p[2] for p in points]
    max_count = max(counts) if counts else 1
    for ia, ib, count, kind in points:
        ax.scatter(ib, ia, s=(30 + 55 * count) * scale ** 2,
                   color=TYPE_COLORS[kind], edgecolor="white",
                   linewidth=0.6 * scale, alpha=0.9, zorder=4)
        ax.text(ib, ia, str(int(count)), ha="center", va="center",
                fontsize=6.2 * scale * BOOST, color="white", fontweight="bold",
                family=FONT, zorder=5)
    ax.set_xlim(-0.6, len(keys_b) - 0.4)
    ax.set_ylim(len(keys_a) - 0.4, -0.6)
    ax.set_xticks(range(len(keys_b)))
    ax.set_xticklabels([label_map.get(k, str(k)) for k in keys_b],
                       rotation=45, ha="right", fontsize=6.4 * scale * BOOST,
                       family=FONT)
    ax.set_yticks(range(len(keys_a)))
    ax.set_yticklabels([label_map.get(k, str(k)) for k in keys_a],
                       fontsize=6.4 * scale * BOOST, family=FONT)
    ax.set_xlabel(f"Chain {result.chain_b.strip()} residues", fontsize=7.6 * scale * BOOST,
                  color=MUTED, family=FONT, labelpad=6)
    ax.set_ylabel(f"Chain {result.chain_a.strip()} residues", fontsize=7.6 * scale * BOOST,
                  color=MUTED, family=FONT, labelpad=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    _panel(ax, "Residue–residue interaction matrix", scale)

    present = [k for k in TYPE_ORDER if any(p[3] == k for p in points)]
    handles = [Patch(facecolor=TYPE_COLORS[k], edgecolor="none",
                     label=TYPE_LABELS[k]) for k in present]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=6.5 * scale * BOOST, title="Dominant type",
              title_fontsize=6.8 * scale * BOOST * BOOST)
    _titles(fig, "Protein–Protein Contact Matrix",
            f"PDB {result.structure.pdb_id}   ·   chains "
            f"{result.chain_a.strip()}/{result.chain_b.strip()}   ·   "
            f"bubble size = interaction records", scale)
    fig.text(0.985, 0.015,
             "Top 12 residues per chain are shown; bubble labels are interaction records.",
             ha="right", va="bottom", fontsize=6.2 * scale * BOOST, color="#9AA4AE",
             family=FONT)
    return fig


def build_dashboard_figure(result: PLResult | PPResult,
                           assessment: PLAssessment | PPAssessment,
                           scale: float = 1.0, dpi: float = 110) -> Figure:
    fig = _figure(10.4, 5.3, scale, dpi)
    counts = {
        k: sum(1 for it in result.interactions if it.kind == k)
        for k in TYPE_ORDER
    }
    records = len(result.interactions)
    if isinstance(result, PLResult):
        d = assessment.descriptors
        tiles = [
            ("BSA (Å²)", f"{assessment.bsa:.0f}", "#2E7D5A"),
            ("Ligand burial", f"{100 * d.burial_fraction:.0f}%", "#C0604A"),
            ("H-bonds", f"{counts['hbond']}", TYPE_COLORS["hbond"]),
            ("Hydrophobic", f"{counts['hydrophobic']}", TYPE_COLORS["hydrophobic"]),
            ("Pocket residues", f"{len(result.nodes)}", "#3D6D96"),
            ("Records / 100 Å²", f"{assessment.interaction_density:.2f}", "#6A5ACD"),
        ]
        subtitle = (f"PDB {result.structure.pdb_id}   ·   ligand "
                    f"{result.ligand.resname} {result.ligand.resseq}   ·   "
                    f"{records} interaction records")
    else:
        tiles = [
            ("BSA (Å²)", f"{assessment.bsa:.0f}", "#2E7D5A"),
            ("Total buried", f"{assessment.total_buried:.0f}", "#3D6D96"),
            ("H-bonds", f"{counts['hbond']}", TYPE_COLORS["hbond"]),
            ("Salt bridges", f"{counts['salt']}", TYPE_COLORS["salt"]),
            ("Interface residues",
             f"{len(result.interface_a)}+{len(result.interface_b)}", "#3F7A4F"),
            ("Records / 100 Å²", f"{assessment.interaction_density:.2f}", "#6A5ACD"),
        ]
        subtitle = (f"PDB {result.structure.pdb_id}   ·   chains "
                    f"{result.chain_a.strip()}/{result.chain_b.strip()}   ·   "
                    f"{records} interaction records")

    tile_ax = fig.add_axes([0.04, 0.62, 0.92, 0.20])
    tile_ax.set_xlim(0, 1)
    tile_ax.set_ylim(0, 1)
    tile_ax.axis("off")
    n = len(tiles)
    width = 0.92 / n
    for i, (label, value, color) in enumerate(tiles):
        x = i * width + 0.01
        tile_ax.add_patch(FancyBboxPatch(
            (x, 0.08), width - 0.02, 0.84,
            boxstyle="round,pad=0.006,rounding_size=0.03",
            transform=tile_ax.transAxes, facecolor=SOFT,
            edgecolor=BORDER, linewidth=0.7 * scale,
        ))
        tile_ax.text(x + (width - 0.02) / 2, 0.60, value, ha="center",
                     va="center", fontsize=15.5 * scale * BOOST, fontweight="bold",
                     color=color, family=FONT, transform=tile_ax.transAxes)
        tile_ax.text(x + (width - 0.02) / 2, 0.24, label, ha="center",
                     va="center", fontsize=6.6 * scale * BOOST, color=MUTED,
                     family=FONT, transform=tile_ax.transAxes)

    donut_ax = fig.add_axes([0.06, 0.12, 0.36, 0.40])
    present = [k for k in TYPE_ORDER if counts[k] > 0]
    if present:
        vals = [counts[k] for k in present]
        colors = [TYPE_COLORS[k] for k in present]
        wedges, _ = donut_ax.pie(
            vals, colors=colors, startangle=90, counterclock=False,
            wedgeprops=dict(width=0.36, edgecolor="white",
                            linewidth=1.4 * scale),
        )
        donut_ax.text(0, 0.06, str(records), ha="center", va="center",
                      fontsize=17 * scale * BOOST, fontweight="bold", color=INK,
                      family=FONT)
        donut_ax.text(0, -0.16, "records", ha="center", va="center",
                      fontsize=6.8 * scale * BOOST, color=MUTED, family=FONT)
        donut_ax.legend(
            wedges, [f"{TYPE_LABELS[k]}  {counts[k]}" for k in present],
            loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
            frameon=False, fontsize=6.2 * scale * BOOST,
        )
    else:
        donut_ax.axis("off")
        donut_ax.text(0.5, 0.5, "No interactions", ha="center",
                      color=MUTED, family=FONT, fontsize=9 * scale * BOOST)
    donut_ax.set_title("Interaction composition", loc="left",
                       fontsize=9.4 * scale * BOOST, fontweight="bold", color=INK,
                       family=FONT, pad=6)

    hot_ax = fig.add_axes([0.56, 0.14, 0.39, 0.36])
    hotspots = assessment.hotspots[:5]
    if hotspots:
        labels = [h["label"] for h in hotspots][::-1]
        values = [h["total"] for h in hotspots][::-1]
        bsa_vals = [h.get("bsa", 0) for h in hotspots][::-1]
        y = np.arange(len(values))
        hot_ax.barh(y, values, color="#2E7D5A", height=0.55,
                    edgecolor="white", linewidth=0.5 * scale)
        for yi, value, bsa in zip(y, values, bsa_vals):
            text = f"{value}"
            if bsa:
                text += f"  ({bsa:.0f} Å²)"
            hot_ax.text(value + max(values) * 0.03, yi, text, va="center",
                        fontsize=6.3 * scale * BOOST, color=INK, family=FONT)
        hot_ax.set_yticks(y)
        hot_ax.set_yticklabels(labels, fontsize=6.8 * scale * BOOST, family=FONT)
        hot_ax.set_xlim(0, max(values) * 1.45)
        hot_ax.set_xlabel("Interaction records", fontsize=7.2 * scale * BOOST,
                          color=MUTED, family=FONT)
        hot_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _clean_axes(hot_ax, scale)
    else:
        hot_ax.axis("off")
        hot_ax.text(0.5, 0.5, "No hotspots", ha="center", color=MUTED,
                    family=FONT, fontsize=9 * scale * BOOST)
    hot_ax.set_title("Top hotspot residues", loc="left",
                     fontsize=9.4 * scale * BOOST, fontweight="bold", color=INK,
                     family=FONT, pad=6)

    _titles(fig, "Interaction Dashboard", subtitle, scale)
    fig.text(0.985, 0.015, "Descriptive statistics; not a binding-affinity prediction.",
             ha="right", va="bottom", fontsize=6.2 * scale * BOOST, color="#9AA4AE",
             family=FONT)
    return fig

def build_hotspot_figure(result: PLResult | PPResult,
                         assessment: PLAssessment | PPAssessment,
                         scale: float = 1.0, dpi: float = 110) -> Figure:
    """Buried area versus interaction count, highlighting hotspot residues."""
    stats = [s for s in result.residue_stats if s.get("total", 0) > 0 and s.get("key")]
    if not stats:
        fig = _figure(7.2, 3.2, scale, dpi)
        fig.text(0.5, 0.5, "No residues with interactions", ha="center",
                 color=MUTED, family=FONT, fontsize=10 * scale * BOOST)
        return fig
    if isinstance(result, PLResult):
        bsa_map = assessment.pocket_residue_bsa
    else:
        bsa_map = {**assessment.residue_bsa_a, **assessment.residue_bsa_b}
    residues = {r.key: r for r in result.structure.residues}
    points = []
    for s in stats:
        key = s["key"]
        bsa = bsa_map.get(key, 0.0)
        if bsa <= 0.05:
            continue
        if isinstance(result, PLResult):
            res = residues.get(key)
            color = CLASS_COLORS[residue_class(res.resname)] if res else "#8A9A90"
        else:
            color = CHAIN_A if key in assessment.residue_bsa_a else CHAIN_B
        points.append({"label": s["label"], "x": bsa, "y": s["total"],
                       "contacts": s.get("contacts", 0), "color": color})
    if not points:
        fig = _figure(7.2, 3.2, scale, dpi)
        fig.text(0.5, 0.5, "No per-residue burial detected", ha="center",
                 color=MUTED, family=FONT, fontsize=10 * scale * BOOST)
        return fig

    fig = _figure(10.2, 5.2, scale, dpi)
    ax = fig.add_axes([0.12, 0.16, 0.70, 0.64])
    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])
    contacts = np.array([p["contacts"] for p in points])
    sizes = (34 + 11 * contacts) * scale ** 2
    colors = [p["color"] for p in points]
    ax.scatter(xs, ys, s=sizes, c=colors, edgecolor="white",
               linewidth=0.8 * scale, alpha=0.9, zorder=4)
    order = sorted(range(len(points)), key=lambda i: -ys[i])[:8]
    offsets = [(8, 8), (8, -15), (-8, 8), (-8, -15), (14, 0), (-14, 0)]
    for rank, i in enumerate(order):
        p = points[i]
        dx, dy = offsets[rank % len(offsets)]
        if p["x"] > max(xs) * 0.82 and dx > 0:
            dx = -dx
        elif p["x"] < max(xs) * 0.10 and dx < 0:
            dx = -dx
        ax.annotate(p["label"], (p["x"], p["y"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.4 * scale * BOOST,
                    color=INK, family=FONT, zorder=5,
                    ha="left" if dx >= 0 else "right",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.85))
    med_x = float(np.median(xs)); med_y = float(np.median(ys))
    ax.axvline(med_x, color="#C5D1C9", linewidth=0.8 * scale,
               linestyle=(0, (2, 2)), zorder=1)
    ax.axhline(med_y, color="#C5D1C9", linewidth=0.8 * scale,
               linestyle=(0, (2, 2)), zorder=1)
    ax.text(0.99, 0.03, "Buried · few records", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.8 * scale * BOOST,
            color="#8C9A92", family=FONT)
    ax.text(0.01, 0.03, "Low burial · low engagement", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=5.8 * scale * BOOST,
            color="#8C9A92", family=FONT)
    ax.text(0.99, 0.96, "Buried · engaged", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8 * scale * BOOST,
            color="#8C9A92", family=FONT)
    ax.text(0.01, 0.96, "Engaged · low burial", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.8 * scale * BOOST,
            color="#8C9A92", family=FONT)
    ax.set_xlabel("Buried surface area per residue (Å²)",
                  fontsize=8.0 * scale * BOOST, color=MUTED, family=FONT)
    ax.set_ylabel("Interaction records", fontsize=8.0 * scale * BOOST,
                  color=MUTED, family=FONT)
    ax.set_xlim(0, max(xs) * 1.18 if len(xs) else 1)
    ax.set_ylim(0, max(ys) * 1.22 if len(ys) else 1)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _clean_axes(ax, scale)
    ax.grid(axis="y", color=GRID, linewidth=0.6 * scale)
    _panel(ax, "Hotspot landscape", scale)

    if isinstance(result, PLResult):
        present = [c for c in CLASS_ORDER
                   if any(p["color"] == CLASS_COLORS[c] for p in points)]
        handles = [Patch(facecolor=CLASS_COLORS[c], edgecolor="none",
                         label=c) for c in present]
    else:
        handles = [Patch(facecolor=CHAIN_A, edgecolor="none",
                         label=f"Chain {result.chain_a.strip()}"),
                   Patch(facecolor=CHAIN_B, edgecolor="none",
                         label=f"Chain {result.chain_b.strip()}")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=7.0 * scale * BOOST,
              title="Residue class" if isinstance(result, PLResult) else "Chain",
              title_fontsize=7.2 * scale * BOOST)
    fig.text(0.985, 0.025, "Bubble area = hydrophobic atom-pair contacts.",
             ha="right", va="bottom", fontsize=5.6 * scale * BOOST,
             color="#9AA4AE", family=FONT)
    _titles(fig, "Hotspot Landscape",
            f"PDB {result.structure.pdb_id}   ·   BSA versus interaction count", scale)
    return fig


def build_enrichment_figure(result: PLResult | PPResult,
                            assessment: PLAssessment | PPAssessment,
                            scale: float = 1.0, dpi: float = 110) -> Figure:
    """Percentage-point enrichment of residue classes relative to background."""
    if isinstance(result, PLResult):
        comp = assessment.pocket_composition
        background = assessment.background_composition
        label = "Pocket"
    else:
        comp = assessment.composition_all
        background = assessment.background_composition
        label = "Interface"
    fig = _figure(10.0, 4.7, scale, dpi)
    ax = fig.add_axes([0.20, 0.20, 0.74, 0.58])
    diffs = [comp.percentages[c] - background.percentages[c] for c in CLASS_ORDER]
    y = np.arange(len(CLASS_ORDER))
    bars = ax.barh(y, diffs, color=[CLASS_COLORS[c] for c in CLASS_ORDER],
                   height=0.62, edgecolor="white", linewidth=0.6 * scale)
    ax.axvline(0, color="#7A8A80", linewidth=0.8 * scale, zorder=2)
    span = max(max(abs(d) for d in diffs), 1.0)
    for yi, cls, diff in zip(y, CLASS_ORDER, diffs):
        obs = comp.percentages[cls]
        bg = background.percentages[cls]
        n = comp.counts[cls]
        text = f"{diff:+.0f} pp   ({obs:.0f}% / {bg:.0f}%, n={n})"
        x = diff + (span * 0.025 if diff >= 0 else -span * 0.025)
        ax.text(x, yi, text, va="center",
                ha="left" if diff >= 0 else "right",
                fontsize=6.6 * scale * BOOST, color=INK, family=FONT)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{CLASS_EN[c]}" for c in CLASS_ORDER],
                       fontsize=7.6 * scale * BOOST, family=FONT)
    ax.invert_yaxis()
    ax.set_xlim(-span * 2.45, span * 2.45)
    ax.set_xlabel("Difference from protein background (percentage points)",
                  fontsize=8.0 * scale * BOOST, color=MUTED, family=FONT)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    _clean_axes(ax, scale)
    _panel(ax, f"{label} residue-class enrichment", scale)
    fig.text(0.985, 0.03,
             "Percentage-point differences are descriptive; small counts should be interpreted cautiously.",
             ha="right", va="bottom", fontsize=6.4 * scale * BOOST,
             color="#9AA4AE", family=FONT)
    _titles(fig, "Residue-Class Enrichment",
            f"PDB {result.structure.pdb_id}   ·   {label.lower()} versus whole-protein background", scale)
    return fig


def build_network_figure(result: PLResult | PPResult, scale: float = 1.0,
                         dpi: float = 110) -> Figure:
    """Circular interaction network: residue nodes and type-colored edges."""
    if not result.interactions:
        fig = _figure(7.2, 3.2, scale, dpi)
        fig.text(0.5, 0.5, "No interactions to draw", ha="center",
                 color=MUTED, family=FONT, fontsize=10 * scale * BOOST)
        return fig
    fig = _figure(8.6, 7.8, scale, dpi)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.79])
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="#E3EAE5",
                        linewidth=0.8 * scale, zorder=0))

    def edge(x1, y1, x2, y2, kind, count, rad):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-", color=TYPE_COLORS[kind],
            linewidth=(0.55 + 0.42 * count) * scale, alpha=0.62,
            shrinkA=8 * scale, shrinkB=8 * scale, zorder=2,
        ))

    present = []
    if isinstance(result, PLResult):
        by_res = {}
        for it in result.interactions:
            entry = by_res.setdefault(it.res_a.key, {"label": it.res_a.label,
                                                     "total": 0, "kinds": {}})
            entry["total"] += 1
            entry["kinds"][it.kind] = entry["kinds"].get(it.kind, 0) + 1
        items = sorted(by_res.items(), key=lambda kv: -kv[1]["total"])[:10]
        n = len(items)
        angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, n, endpoint=False)
        residues = {r.key: r for r in result.structure.residues}
        for idx, (key, entry) in enumerate(items):
            if idx > 0:
                prev = items[idx - 1][1]["kinds"]
            theta = float(angles[idx])
            x, y = np.cos(theta), np.sin(theta)
            kind = _dominant(entry["kinds"])
            edge(x, y, 0.0, 0.0, kind, entry["total"], 0.10)
            res = residues.get(key)
            color = CLASS_COLORS[residue_class(res.resname)] if res else "#8A9A90"
            radius = 0.035 + 0.004 * entry["total"]
            ax.add_patch(Circle((x, y), radius, facecolor=color,
                                edgecolor="white", linewidth=0.6 * scale, zorder=5))
            ha = "left" if x >= 0 else "right"
            ax.text(x * 1.13, y * 1.13, entry["label"], ha=ha, va="center",
                    fontsize=7.4 * scale * BOOST, color=INK, family=FONT, zorder=6)
            if entry["kinds"]:
                present.extend(entry["kinds"].keys())
        ax.add_patch(Circle((0, 0), 0.13, facecolor=PRIMARY, edgecolor="white",
                            linewidth=1.0 * scale, zorder=6))
        lig = result.ligand
        ax.text(0, 0, f"{lig.resname}\n{lig.resseq}", ha="center", va="center",
                fontsize=8.0 * scale * BOOST, color="white", family=FONT,
                fontweight="bold", zorder=7)
        subtitle = (f"PDB {result.structure.pdb_id}   ·   ligand "
                    f"{lig.resname} {lig.resseq}   ·   nodes = residues")
    else:
        by_res_a = {}
        by_res_b = {}
        groups = {}
        for it in result.interactions:
            if it.res_b is None:
                continue
            ea = by_res_a.setdefault(it.res_a.key, {"label": it.res_a.label,
                                                    "total": 0})
            ea["total"] += 1
            eb = by_res_b.setdefault(it.res_b.key, {"label": it.res_b.label,
                                                    "total": 0})
            eb["total"] += 1
            g = groups.setdefault((it.res_a.key, it.res_b.key),
                                  {"count": 0, "kinds": {}})
            g["count"] += 1
            g["kinds"][it.kind] = g["kinds"].get(it.kind, 0) + 1
        items_a = sorted(by_res_a.items(), key=lambda kv: -kv[1]["total"])[:8]
        items_b = sorted(by_res_b.items(), key=lambda kv: -kv[1]["total"])[:8]
        pos = {}
        if items_a:
            ang_a = np.deg2rad(np.linspace(105, 255, len(items_a)))
            for (key, entry), theta in zip(items_a, ang_a):
                pos[key] = (float(np.cos(theta)), float(np.sin(theta)), entry)
        if items_b:
            ang_b = np.deg2rad(np.linspace(-75, 75, len(items_b)))
            for (key, entry), theta in zip(items_b, ang_b):
                pos[key] = (float(np.cos(theta)), float(np.sin(theta)), entry)
        for (ka, kb), entry in groups.items():
            if ka not in pos or kb not in pos:
                continue
            x1, y1, _ = pos[ka]
            x2, y2, _ = pos[kb]
            edge(x1, y1, x2, y2, _dominant(entry["kinds"]),
                 entry["count"], 0.08)
            present.extend(k for k in entry["kinds"] if k not in present)
        for side, side_items in (("A", items_a), ("B", items_b)):
            for key, entry in side_items:
                x, y, _ = pos[key]
                radius = 0.035 + 0.0035 * entry["total"]
                color = CHAIN_A if side == "A" else CHAIN_B
                ax.add_patch(Circle((x, y), radius, facecolor=color,
                                    edgecolor="white", linewidth=0.6 * scale,
                                    zorder=5))
                ha = "right" if x < 0 else "left"
                ax.text(x * 1.12, y * 1.12, entry["label"], ha=ha, va="center",
                        fontsize=7.4 * scale * BOOST, color=INK, family=FONT,
                        zorder=6)
        ax.text(-0.78, 1.20, f"Chain {result.chain_a.strip()}", ha="center",
                va="center", fontsize=10.0 * scale * BOOST, color=CHAIN_A,
                fontweight="bold", family=FONT)
        ax.text(0.78, 1.20, f"Chain {result.chain_b.strip()}", ha="center",
                va="center", fontsize=10.0 * scale * BOOST, color=CHAIN_B,
                fontweight="bold", family=FONT)
        subtitle = (f"PDB {result.structure.pdb_id}   ·   chains "
                    f"{result.chain_a.strip()}/{result.chain_b.strip()}   ·   "
                    f"nodes = interface residues")

    present = [k for k in TYPE_ORDER if k in set(present)]
    handles = [Patch(facecolor=TYPE_COLORS[k], edgecolor="none",
                     label=TYPE_LABELS[k]) for k in present]
    if handles:
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03),
                  ncol=min(4, len(handles)), frameon=False,
                  fontsize=7.0 * scale * BOOST, handlelength=1.2)
    _titles(fig, "Interaction Network",
            subtitle + "   ·   node size ∝ interaction records", scale)
    fig.text(0.985, 0.015,
             "Node colour = residue class; edge width ∝ records; colour = dominant interaction type."
             if isinstance(result, PLResult) else
             "Node colour = chain; edge width ∝ records; colour = dominant interaction type.",
             ha="right", va="bottom", fontsize=6.4 * scale * BOOST,
             color="#9AA4AE", family=FONT)
    return fig

def build_type_class_figure(result: PLResult | PPResult,
                            assessment: PLAssessment | PPAssessment,
                            scale: float = 1.0, dpi: float = 110) -> Figure:
    """Heatmap of interaction types by residue class."""
    nodes = result.nodes if isinstance(result, PLResult) else (result.nodes_a + result.nodes_b)
    matrix = np.zeros((len(CLASS_ORDER), len(TYPE_ORDER)))
    for node in nodes:
        cls = residue_class(node.residue.resname)
        ci = CLASS_ORDER.index(cls)
        for it in node.interactions:
            ti = TYPE_ORDER.index(it.kind)
            matrix[ci, ti] += 1
    if matrix.sum() == 0:
        fig = _figure(7.2, 3.2, scale, dpi)
        fig.text(0.5, 0.5, "No interactions for class analysis", ha="center",
                 color=MUTED, family=FONT, fontsize=10 * scale * BOOST)
        return fig
    present_types = [i for i in range(len(TYPE_ORDER)) if matrix[:, i].sum() > 0]
    present_classes = [i for i in range(len(CLASS_ORDER)) if matrix[i, :].sum() > 0]
    mat = matrix[np.ix_(present_classes, present_types)]
    fig = _figure(10.2, max(4.6, 0.42 * len(present_classes) + 2.0), scale, dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.0], wspace=0.06,
                          left=0.18, right=0.96, top=0.81, bottom=0.20)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "tc", ["#FFFFFF", "#E5F0E9", "#8DBFA2", "#2E7D5A", "#174534"])
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=max(1, mat.max()))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > mat.max() * 0.55 else INK,
                        fontsize=7.8 * scale * BOOST, fontweight="bold", family=FONT)
            else:
                ax.text(j, i, "·", ha="center", va="center", color="#A6B2AA",
                        fontsize=7.2 * scale * BOOST, family=FONT)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([TYPE_LABELS[TYPE_ORDER[i]].replace(" ", "\n", 1)
                        for i in present_types], fontsize=7.4 * scale * BOOST,
                       family=FONT)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([CLASS_EN[CLASS_ORDER[i]] for i in present_classes],
                       fontsize=7.8 * scale * BOOST, family=FONT)
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4 * scale)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Interaction type", fontsize=8.2 * scale * BOOST,
                  color=MUTED, family=FONT)
    _panel(ax, "Interaction type × residue class", scale)

    totals = mat.sum(axis=1)
    ax2.barh(np.arange(mat.shape[0]), totals,
             color=[CLASS_COLORS[CLASS_ORDER[i]] for i in present_classes],
             height=0.62, edgecolor="white", linewidth=0.6 * scale)
    for i, total in enumerate(totals):
        ax2.text(total + max(totals) * 0.03, i, f"{int(total)}", va="center",
                 ha="left", fontsize=7.0 * scale * BOOST, color=INK, family=FONT)
    ax2.set_xlim(0, max(totals) * 1.22)
    ax2.set_xlabel("Involvements", fontsize=7.8 * scale * BOOST,
                   color=MUTED, family=FONT)
    ax2.tick_params(axis="y", left=False, labelleft=False)
    _clean_axes(ax2, scale)
    ax2.set_title("Total involvements", loc="left", fontsize=9.4 * scale * BOOST,
                  fontweight="bold", color=INK, family=FONT, pad=8)
    note = ("Each interaction can involve two residues in protein–protein mode; "
            "counts are residue–interaction involvements."
            if isinstance(result, PPResult) else
            "Counts are residue–interaction involvements within the pocket.")
    fig.text(0.985, 0.02, note, ha="right", va="bottom",
             fontsize=6.4 * scale * BOOST, color="#9AA4AE", family=FONT)
    _titles(fig, "Interaction Type × Residue Class",
            f"PDB {result.structure.pdb_id}   ·   chemical preference of interaction types", scale)
    return fig


def build_hbond_geometry_figure(result: PLResult | PPResult, scale: float = 1.0,
                                dpi: float = 110) -> Figure:
    hb = [it for it in result.interactions if it.kind == "hbond"]
    if not hb:
        fig = _figure(7.2, 3.2, scale, dpi)
        fig.text(0.5, 0.5, "No hydrogen bonds detected", ha="center",
                 color=MUTED, family=FONT, fontsize=10 * scale * BOOST)
        return fig
    dist = np.array([it.distance for it in hb])
    ang = np.array([it.angle if it.angle is not None else 0 for it in hb])
    fig = _figure(7.6, 6.7, scale, dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[4.2, 1.0], height_ratios=[1.0, 4.2],
                          wspace=0.04, hspace=0.04,
                          left=0.15, right=0.94, top=0.82, bottom=0.17)
    ax = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=ax)
    right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax.axvspan(2.4, 3.5, ymin=0, ymax=1, color="#EAF3EC", zorder=0)
    ax.axhspan(120, 180, xmin=0, xmax=1, color="#EAF3EC", zorder=0, alpha=0.7)
    ax.scatter(dist, ang, s=42 * scale ** 2, color=TYPE_COLORS["hbond"],
               edgecolor="white", linewidth=0.7 * scale, alpha=0.92, zorder=4)
    ax.axvline(3.5, color="#9AB3A3", linewidth=0.8 * scale,
               linestyle=(0, (2, 2)), zorder=2)
    ax.axhline(120, color="#9AB3A3", linewidth=0.8 * scale,
               linestyle=(0, (2, 2)), zorder=2)
    md, ma = float(np.median(dist)), float(np.median(ang))
    ax.axvline(md, color="#2E7D32", linewidth=0.9 * scale,
               linestyle=(0, (4, 2)), zorder=3)
    ax.axhline(ma, color="#2E7D32", linewidth=0.9 * scale,
               linestyle=(0, (4, 2)), zorder=3)
    ax.text(0.02, 0.96, f"n = {len(hb)}\nmedian {md:.2f} Å / {ma:.0f}°",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.2 * scale * BOOST, color=MUTED, family=FONT,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.text(3.52, 170, "3.5 Å", fontsize=6.8 * scale * BOOST,
            color=MUTED, family=FONT)
    ax.text(2.45, 123, "120°", fontsize=6.8 * scale * BOOST,
            color=MUTED, family=FONT)
    ax.set_xlim(max(2.4, dist.min() - 0.15), max(4.2, dist.max() + 0.15))
    ax.set_ylim(70, 180)
    ax.set_xlabel("Donor–acceptor distance (Å)", fontsize=8.4 * scale * BOOST,
                  color=MUTED, family=FONT)
    ax.set_ylabel("D–H···A angle (°)", fontsize=8.4 * scale * BOOST,
                  color=MUTED, family=FONT)
    ax.grid(color=GRID, linewidth=0.5 * scale, zorder=0)
    _clean_axes(ax, scale)

    bins_d = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10)
    top.hist(dist, bins=bins_d, color="#B7D3BE", edgecolor="white",
             linewidth=0.6 * scale)
    top.axvline(md, color="#2E7D32", linewidth=0.9 * scale,
                linestyle=(0, (4, 2)))
    top.set_ylabel("Count", fontsize=6.8 * scale * BOOST, color=MUTED, family=FONT)
    top.tick_params(axis="x", labelbottom=False)
    top.tick_params(axis="y", labelsize=6.4 * scale * BOOST)
    for spine in ("top", "right"):
        top.spines[spine].set_visible(False)
    top.spines["left"].set_color(GRID)
    top.spines["bottom"].set_color(GRID)

    bins_a = np.linspace(70, 180, 10)
    right.hist(ang, bins=bins_a, orientation="horizontal",
               color="#B7D3BE", edgecolor="white", linewidth=0.6 * scale)
    right.axhline(ma, color="#2E7D32", linewidth=0.9 * scale,
                  linestyle=(0, (4, 2)))
    right.set_xlabel("Count", fontsize=6.8 * scale * BOOST, color=MUTED, family=FONT)
    right.tick_params(axis="y", labelleft=False)
    right.tick_params(axis="x", labelsize=6.4 * scale * BOOST)
    for spine in ("top", "right"):
        right.spines[spine].set_visible(False)
    right.spines["left"].set_color(GRID)
    right.spines["bottom"].set_color(GRID)

    _titles(fig, "Hydrogen-Bond Geometry",
            f"PDB {result.structure.pdb_id}   ·   distance and angle distributions", scale)
    fig.text(0.985, 0.015,
             "Shaded regions indicate the commonly favourable distance/angle ranges.",
             ha="right", va="bottom", fontsize=6.4 * scale * BOOST,
             color="#9AA4AE", family=FONT)
    return fig


def build_distance_stats_figure(result: PLResult | PPResult, scale: float = 1.0,
                                dpi: float = 110) -> Figure:
    present = [k for k in TYPE_ORDER
               if any(it.kind == k for it in result.interactions)]
    if not present:
        fig = _figure(7.2, 3.2, scale, dpi)
        fig.text(0.5, 0.5, "No interaction distances", ha="center",
                 color=MUTED, family=FONT, fontsize=10 * scale * BOOST)
        return fig
    data = {k: np.array([it.distance for it in result.interactions if it.kind == k])
            for k in present}
    fig = _figure(10.2, 5.1, scale, dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.38,
                          left=0.17, right=0.96, top=0.81, bottom=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, kind in enumerate(present):
        vals = data[kind]
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        lo, hi = float(vals.min()), float(vals.max())
        ax1.add_patch(Rectangle((q1, idx - 0.20), max(q3 - q1, 0.01), 0.40,
                                facecolor=TYPE_COLORS[kind], alpha=0.28,
                                edgecolor=TYPE_COLORS[kind], linewidth=0.8 * scale))
        ax1.plot([lo, hi], [idx, idx], color=TYPE_COLORS[kind],
                 linewidth=0.8 * scale, zorder=2)
        ax1.plot([med, med], [idx - 0.26, idx + 0.26], color=TYPE_COLORS[kind],
                 linewidth=1.6 * scale, zorder=3)
        jitter = (np.arange(len(vals)) % 7 - 3) * 0.02
        ax1.scatter(vals, np.full(len(vals), idx) + jitter, s=24 * scale ** 2,
                    color=TYPE_COLORS[kind], edgecolor="white",
                    linewidth=0.4 * scale, zorder=4)
        ax1.text(max(vals) + 0.08, idx, f"{med:.2f} (n={len(vals)})",
                 va="center", fontsize=6.5 * scale * BOOST, color=INK,
                 family=FONT)
    ax1.set_yticks(range(len(present)))
    ax1.set_yticklabels([TYPE_LABELS[k] for k in present],
                        fontsize=7.6 * scale * BOOST, family=FONT)
    ax1.set_xlabel("Distance (Å)", fontsize=8.2 * scale * BOOST,
                   color=MUTED, family=FONT)
    ax1.set_xlim(left=2.3)
    _clean_axes(ax1, scale)
    ax1.grid(axis="x", color=GRID, linewidth=0.6 * scale)
    _panel(ax1, "Distance distributions", scale)

    for kind in present:
        vals = np.sort(data[kind])
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax2.step(vals, y, where="post", color=TYPE_COLORS[kind],
                 linewidth=1.3 * scale, label=f"{TYPE_LABELS[kind]} (n={len(vals)})")
        ax2.scatter(vals, y, s=14 * scale ** 2, color=TYPE_COLORS[kind],
                    edgecolor="white", linewidth=0.3 * scale, zorder=4)
    ax2.set_xlabel("Distance (Å)", fontsize=8.2 * scale * BOOST,
                   color=MUTED, family=FONT)
    ax2.set_ylabel("Cumulative fraction", fontsize=8.2 * scale * BOOST,
                   color=MUTED, family=FONT)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(left=2.3)
    ax2.legend(frameon=False, fontsize=6.6 * scale * BOOST,
               loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
               handlelength=1.2, columnspacing=1.2)
    _clean_axes(ax2, scale)
    ax2.grid(axis="y", color=GRID, linewidth=0.6 * scale)
    _panel(ax2, "Cumulative distance distributions", scale)
    _titles(fig, "Interaction Distance Statistics",
            f"PDB {result.structure.pdb_id}   ·   {len(result.interactions)} interaction records", scale)
    return fig


def build_ligand_burial_map_figure(result: PLResult, assessment: PLAssessment,
                                   scale: float = 1.0, dpi: float = 110) -> Figure:
    from .render import _make_basis
    ligand = result.ligand
    chem = result.ligand_chem
    atoms = chem.atoms
    coords = np.array([a.coord for a in atoms], dtype=float)
    center, basis = _make_basis(coords)
    uv = (coords - center) @ basis.T
    span = max(uv.max(axis=0) - uv.min(axis=0))
    uv = uv / max(span, 1e-6)
    buried = assessment.sasa.per_atom_buried_b
    free_per_atom = buried + assessment.sasa.per_atom_complex_b
    frac = np.divide(buried, free_per_atom, out=np.zeros_like(buried),
                     where=free_per_atom > 1e-6)
    contact_atoms = set()
    for it in result.interactions:
        if it.atom_b and it.atom_b != "ring":
            contact_atoms.add(it.atom_b)
    fig = _figure(7.8, 6.9, scale, dpi)
    ax = fig.add_axes([0.05, 0.13, 0.88, 0.73])
    ax.set_aspect("equal")
    ax.axis("off")
    for i, j in chem.bonds:
        ax.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]],
                color="#4A5652", linewidth=2.2 * scale, solid_capstyle="round",
                zorder=2)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "burial", ["#F7FCF5", "#D5EBD7", "#7CBF8A", "#2E7D5A", "#174534"])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    for i, atom in enumerate(atoms):
        color = cmap(norm(float(frac[i])))
        r = 0.032 if atom.element.upper() != "C" else 0.024
        is_contact = atom.name in contact_atoms
        ax.scatter([uv[i, 0]], [uv[i, 1]], s=(r * 900) * scale ** 2,
                   color=color, edgecolor="#C8A24A" if is_contact else "white",
                   linewidth=1.3 * scale if is_contact else 0.5 * scale,
                   zorder=5)
        if atom.element.upper() != "C" and is_contact:
            ax.text(uv[i, 0], uv[i, 1] + 0.065, atom.name, ha="center",
                    va="bottom", fontsize=6.8 * scale * BOOST, color=INK,
                    family=FONT, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.8))
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.20, 0.085, 0.45, 0.022])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Ligand atom burial fraction", fontsize=7.2 * scale * BOOST,
                 color=MUTED, family=FONT)
    cb.ax.tick_params(labelsize=6.4 * scale * BOOST, colors=MUTED)
    _titles(fig, "Ligand Atom Burial Map",
            f"PDB {result.structure.pdb_id}   ·   {ligand.resname} {ligand.resseq}   ·   "
            f"mean per-atom burial {100 * np.mean(frac):.0f}%", scale)
    return fig


def build_pair_type_figure(result: PPResult, scale: float = 1.0,
                           dpi: float = 110) -> Figure:
    groups = {}
    label_map = _residue_labels(result.structure)
    for it in result.interactions:
        if it.res_b is None:
            continue
        key = (it.res_a.key, it.res_b.key)
        entry = groups.setdefault(key, {"count": 0, "kinds": {}})
        entry["count"] += 1
        entry["kinds"][it.kind] = entry["kinds"].get(it.kind, 0) + 1
    if not groups:
        fig = _figure(7.2, 3.2, scale, dpi)
        fig.text(0.5, 0.5, "No residue-pair interactions", ha="center",
                 color=MUTED, family=FONT, fontsize=10 * scale * BOOST)
        return fig
    items = sorted(groups.items(), key=lambda kv: -kv[1]["count"])[:12]
    types = [k for k in TYPE_ORDER if any(e["kinds"].get(k, 0) for _, e in items)]
    mat = np.zeros((len(items), len(types)))
    for i, (_, entry) in enumerate(items):
        for j, kind in enumerate(types):
            mat[i, j] = entry["kinds"].get(kind, 0)
    fig = _figure(10.2, max(5.0, 0.25 * len(items) + 2.0), scale, dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.0], wspace=0.06,
                          left=0.30, right=0.96, top=0.81, bottom=0.21)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "pt", ["#FFFFFF", "#E8F1EA", "#98C1A6", "#2E7D5A", "#174534"])
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=max(1, mat.max()))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > mat.max() * 0.55 else INK,
                        fontsize=7.0 * scale * BOOST, fontweight="bold", family=FONT)
            else:
                ax.text(j, i, "·", ha="center", va="center", color="#A6B2AA",
                        fontsize=6.8 * scale * BOOST, family=FONT)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([TYPE_LABELS[k].replace(" ", "\n", 1) for k in types],
                       fontsize=7.2 * scale * BOOST, family=FONT)
    ax.set_yticks(range(len(items)))
    labels = []
    for (ka, kb), _ in items:
        la = label_map.get(ka, str(ka)); lb = label_map.get(kb, str(kb))
        labels.append(f"{la} — {lb}")
    ax.set_yticklabels(labels, fontsize=6.6 * scale * BOOST, family=FONT)
    ax.set_xticks(np.arange(-0.5, len(types), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(items), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3 * scale)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _panel(ax, "Residue-pair × interaction type", scale)
    totals = mat.sum(axis=1)
    ax2.barh(np.arange(len(items)), totals, color="#7FB591", height=0.62,
             edgecolor="white", linewidth=0.5 * scale)
    for i, total in enumerate(totals):
        ax2.text(total + max(totals) * 0.03, i, f"{int(total)}", va="center",
                 ha="left", fontsize=6.8 * scale * BOOST, color=INK, family=FONT)
    ax2.set_xlim(0, max(totals) * 1.22)
    ax2.set_xlabel("Records", fontsize=7.6 * scale * BOOST, color=MUTED, family=FONT)
    ax2.tick_params(axis="y", left=False, labelleft=False)
    _clean_axes(ax2, scale)
    _panel(ax2, "Pair totals", scale)
    _titles(fig, "Residue-Pair Interaction Composition",
            f"PDB {result.structure.pdb_id}   ·   chains "
            f"{result.chain_a.strip()}/{result.chain_b.strip()}", scale)
    fig.text(0.985, 0.015, "Top 12 residue pairs by interaction records.",
             ha="right", va="bottom", fontsize=6.4 * scale * BOOST,
             color="#9AA4AE", family=FONT)
    return fig
