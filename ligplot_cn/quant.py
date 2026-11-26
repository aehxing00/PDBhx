"""定量分析图：残基互作谱 + 距离分布（英文出版级）。

左图：每个残基的相互作用记录数（按类型堆叠）。
右图：各类型相互作用距离分布（条带图 + 中位数）。
"""

from __future__ import annotations

from statistics import median

import numpy as np
import matplotlib
matplotlib.use("Agg", force=False)
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .analysis import PLResult, PPResult

FONT = "DejaVu Sans"
BOOST = 1.40
COLORS = {
    "hbond": "#2E7D32",
    "salt": "#7B1FA2",
    "pi": "#00796B",
    "metal": "#6D4C41",
    "hydrophobic": "#AE3B2C",
}
LABELS = {
    "hbond": "Hydrogen bond",
    "salt": "Salt bridge",
    "pi": "π–π stacking",
    "metal": "Metal coordination",
    "hydrophobic": "Hydrophobic contact",
}
ORDER = ["hbond", "salt", "pi", "metal", "hydrophobic"]
TEXT = "#1F2933"
MUTED = "#5B6570"
GRID = "#DDE4E0"


def _empty_figure(message: str, scale: float, dpi: float) -> Figure:
    fig = Figure(figsize=(9.2 * scale, 4.6 * scale), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.55, "Interaction Quantification", fontsize=13 * scale * BOOST,
            fontweight="bold", ha="center", color=TEXT, family=FONT)
    ax.text(0.5, 0.43, message, fontsize=9 * scale * BOOST, ha="center",
            color=MUTED, family=FONT)
    return fig


def build_quant_figure(result: PLResult | PPResult, scale: float = 1.0,
                       dpi: float = 110) -> Figure:
    stats = list(result.residue_stats)
    interactions = result.interactions
    if not stats or not interactions:
        return _empty_figure("No interactions detected with the current thresholds.",
                             scale, dpi)

    present = [k for k in ORDER if any(s[k] > 0 for s in stats)]
    total_records = len(interactions)

    if isinstance(result, PLResult):
        ligand = result.ligand
        ligand_text = f"{ligand.resname} {ligand.resseq}{ligand.icode.strip()} (Chain {ligand.chain.strip() or '-'})"
        title = f"Interaction Quantification — {ligand.resname} {ligand.resseq}" \
                f"{ligand.icode.strip()} (Chain {ligand.chain.strip() or '-'})"
        subtitle = (
            f"PDB {result.structure.pdb_id}   ·   Ligand {ligand_text}   ·   "
            f"{len(stats)} contacting residues   ·   {total_records} interaction records"
        )
    else:
        ca = result.chain_a.strip() or "-"
        cb = result.chain_b.strip() or "-"
        title = f"Interaction Quantification — Chain {ca} / Chain {cb}"
        subtitle = (
            f"PDB {result.structure.pdb_id}   ·   Interface residues "
            f"{len(result.interface_a)} + {len(result.interface_b)}   ·   "
            f"{total_records} interaction records"
        )

    # 只保留有互作的残基；数量过多时合并尾部
    stats = [s for s in stats if s["total"] > 0]
    max_rows = 16
    if len(stats) > max_rows:
        head = stats[:max_rows - 1]
        tail = stats[max_rows - 1:]
        merged = {
            "label": f"Other ({len(tail)} residues)",
            "hbond": sum(s["hbond"] for s in tail),
            "salt": sum(s["salt"] for s in tail),
            "pi": sum(s["pi"] for s in tail),
            "metal": sum(s["metal"] for s in tail),
            "hydrophobic": sum(s["hydrophobic"] for s in tail),
            "total": sum(s["total"] for s in tail),
            "min_dist": min(s["min_dist"] for s in tail),
        }
        stats = head + [merged]

    fig_w, fig_h = 10.8, 5.0
    fig = Figure(figsize=(fig_w * scale, fig_h * scale), dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.45, 1.0], wspace=0.55,
        left=0.23, right=0.975, top=0.80, bottom=0.21,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ---- 面板 A：残基互作谱 ----
    n = len(stats)
    y = np.arange(n)[::-1]
    height = 0.62
    left = np.zeros(n)
    for kind in present:
        vals = np.array([s[kind] for s in stats], dtype=float)
        ax1.barh(y, vals, left=left, height=height, color=COLORS[kind],
                 edgecolor="white", linewidth=0.6 * scale, zorder=3)
        left += vals
    totals = np.array([s["total"] for s in stats], dtype=float)
    xmax = max(float(totals.max()), 1.0)
    for yi, s in zip(y, stats):
        ax1.text(s["total"] + 0.06, yi, f"{s['total']}",
                 va="center", ha="left", fontsize=7.0 * scale * BOOST,
                 color=TEXT, fontweight="bold", family=FONT, zorder=5)
        ax1.text(xmax + 1.00, yi, f"{s['min_dist']:.2f} Å",
                 va="center", ha="right", fontsize=6.2 * scale * BOOST,
                 color=MUTED, family=FONT, zorder=5)
    ax1.set_yticks(y)
    ax1.set_yticklabels([s["label"] for s in stats], fontsize=7.6 * scale * BOOST, family=FONT)
    ax1.set_xlim(0, xmax + 1.42)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Interaction records per residue", fontsize=8.0 * scale * BOOST,
                   color=TEXT, family=FONT, labelpad=6)
    ax1.set_title("A   Per-residue interaction profile", loc="left",
                  fontsize=9.6 * scale * BOOST, fontweight="bold", color=TEXT,
                  family=FONT, pad=8)
    ax1.tick_params(axis="x", labelsize=7.0 * scale * BOOST, colors=MUTED)
    ax1.tick_params(axis="y", length=0, pad=4)
    ax1.grid(axis="x", color=GRID, linewidth=0.6 * scale, zorder=0)
    ax1.set_axisbelow(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    ax1.spines["left"].set_color(GRID)
    ax1.spines["bottom"].set_color(GRID)

    # ---- 面板 B：距离分布 ----
    categories = present
    by_type = {k: [it.distance for it in interactions if it.kind == k] for k in categories}
    positions = {k: i for i, k in enumerate(categories)}
    for kind in categories:
        values = by_type[kind]
        if not values:
            continue
        y0 = positions[kind]
        jitter = (np.arange(len(values)) % 7 - 3) * 0.045
        ax2.scatter(values, np.full(len(values), y0) + jitter,
                    s=22 * scale, color=COLORS[kind], edgecolor="white",
                    linewidth=0.4 * scale, alpha=0.9, zorder=4)
        med = median(values)
        ax2.plot([med, med], [y0 - 0.30, y0 + 0.30], color="#424242",
                 linewidth=1.0 * scale, linestyle=(0, (2.0, 1.6)), zorder=5)
        ax2.text(med, y0 + 0.36, f"{med:.2f}", ha="center", va="bottom",
                 fontsize=6.5 * scale * BOOST, color="#424242", family=FONT, zorder=6)
    ax2.set_yticks([positions[k] for k in categories])
    ax2.set_yticklabels(
        [f"{LABELS[k]}\n(n={len(by_type[k])})" for k in categories],
        fontsize=7.2 * scale * BOOST, family=FONT,
    )
    ax2.set_ylim(-0.7, len(categories) - 0.25)
    all_d = [it.distance for it in interactions]
    ax2.set_xlim(max(1.5, min(all_d) - 0.35), max(all_d) + 0.45)
    ax2.set_xlabel("Distance (Å)", fontsize=8.0 * scale * BOOST, color=TEXT,
                   family=FONT, labelpad=6)
    ax2.set_title("B   Distance distribution", loc="left",
                  fontsize=9.6 * scale * BOOST, fontweight="bold", color=TEXT,
                  family=FONT, pad=8)
    ax2.tick_params(axis="x", labelsize=7.0 * scale * BOOST, colors=MUTED)
    ax2.tick_params(axis="y", length=0, pad=4)
    ax2.grid(axis="x", color=GRID, linewidth=0.6 * scale, zorder=0)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.spines["left"].set_color(GRID)
    ax2.spines["bottom"].set_color(GRID)

    # ---- 标题与图例 ----
    fig.text(0.5, 0.985, title, ha="center", va="top",
             fontsize=12.2 * scale * BOOST, fontweight="bold", color=TEXT, family=FONT)
    fig.text(0.5, 0.916, subtitle, ha="center", va="top",
             fontsize=7.2 * scale * BOOST, color=MUTED, family=FONT)
    handles = [Patch(facecolor=COLORS[k], edgecolor="none", label=LABELS[k])
               for k in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=7.2 * scale * BOOST, bbox_to_anchor=(0.5, 0.045),
               handlelength=1.2, columnspacing=1.6, handletextpad=0.5)
    fig.text(0.985, 0.012, "Generated by LigPlot-CN", ha="right", va="bottom",
             fontsize=6.2 * scale * BOOST, color="#9AA4AE", family=FONT)
    return fig
