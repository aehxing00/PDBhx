"""出版级二维示意图绘制。

- 蛋白-小分子：经典二维示意图风格，配体居中、残基环绕，疏水残基用红色“睫毛”弧表示
- 蛋白-蛋白：左右两栏界面残基网络，连线表示不同类型相互作用

图中所有文字均为英文，尺寸单位为英寸，按 scale 缩放用于界面预览。
"""

from __future__ import annotations

from functools import lru_cache
from math import atan2, cos, pi, sin

import numpy as np
import matplotlib
matplotlib.use("Agg", force=False)
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch
from matplotlib.textpath import TextPath

from .analysis import PLResult, PPResult
from .chemdata import element_color, element_radius
from .interactions import RawInteraction

FONT = "DejaVu Sans"

COLORS = {
    "hbond": "#2E7D32",
    "hydrophobic": "#AE3B2C",
    "salt": "#7B1FA2",
    "pi": "#00796B",
    "metal": "#6D4C41",
    "text": "#1F2933",
    "subtext": "#5B6570",
    "ligand_c": "#33383B",
    "chain_a": "#3D6D96",
    "chain_b": "#3F7A4F",
    "panel_a": "#F2F7FC",
    "panel_b": "#F3FAF5",
    "spoke": "#B04636",
}

KIND_LABEL_EN = {
    "hbond": "Hydrogen bond",
    "hydrophobic": "Hydrophobic contact",
    "salt": "Salt bridge",
    "pi": "π–π stacking",
    "metal": "Metal coordination",
}

LEGEND_ORDER = ["hbond", "hydrophobic", "salt", "pi", "metal"]


@lru_cache(maxsize=1024)
def text_extent_in(text: str, fontsize: float, weight: str = "normal") -> tuple[float, float]:
    try:
        prop = FontProperties(family=FONT, size=fontsize, weight=weight)
        bb = TextPath((0, 0), text, prop=prop).get_extents()
        return max(bb.width / 72.0, 0.02), max(bb.height / 72.0, 0.02)
    except Exception:
        em = fontsize / 72.0
        return len(text) * 0.55 * em, 0.75 * em


def _text_width(text: str, fontsize: float, weight: str = "normal") -> float:
    return text_extent_in(text, fontsize, weight)[0]


def _rect_edge(cx, cy, w, h, tx, ty):
    dx, dy = tx - cx, ty - cy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return cx, cy
    sx = (w / 2.0) / abs(dx) if abs(dx) > 1e-9 else float("inf")
    sy = (h / 2.0) / abs(dy) if abs(dy) > 1e-9 else float("inf")
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def _project(coords: np.ndarray, center: np.ndarray, basis: np.ndarray, scale: float) -> np.ndarray:
    return (coords - center) @ basis.T * scale


def _make_basis(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = coords.mean(axis=0)
    centered = coords - center
    if len(coords) < 2:
        return center, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    try:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return center, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    basis = vt[:2]
    if basis.shape[0] < 2:
        fallback = np.array([0.0, 1.0, 0.0])
        second = fallback - np.dot(fallback, basis[0]) * basis[0]
        if np.linalg.norm(second) < 1e-6:
            second = np.array([0.0, 0.0, 1.0])
        basis = np.vstack([basis[0], second / np.linalg.norm(second)])
    return center, basis


# ---------------------------------------------------------------------------
# 通用绘制组件
# ---------------------------------------------------------------------------

def _relax_labels(items: list[dict], iterations: int = 80, margin: float = 0.004) -> None:
    """对标签矩形做简单的两两排斥。"""
    if len(items) < 2:
        return
    for _ in range(iterations):
        moved = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                dx = b["pos"][0] - a["pos"][0]
                dy = b["pos"][1] - a["pos"][1]
                ox = (a["w"] + b["w"]) / 2.0 + margin - abs(dx)
                oy = (a["h"] + b["h"]) / 2.0 + margin - abs(dy)
                if ox > 0 and oy > 0:
                    if ox <= oy:
                        shift = ox / 2.0
                        sign = 1.0 if dx >= 0 else -1.0
                        a["pos"][0] -= shift * sign
                        b["pos"][0] += shift * sign
                    else:
                        shift = oy / 2.0
                        sign = 1.0 if dy >= 0 else -1.0
                        a["pos"][1] -= shift * sign
                        b["pos"][1] += shift * sign
                    moved = True
        if not moved:
            break


def _draw_legend(ax, x0: float, x1: float, y: float, s: float, show_pi: bool,
                 mode: str = "pl", show_metal: bool = False) -> None:
    kinds = [k for k in LEGEND_ORDER
             if (k != "pi" or show_pi) and (k != "metal" or show_metal)]
    fs = 9.2 * s
    entries = []
    for kind in kinds:
        label = KIND_LABEL_EN[kind]
        if kind in ("hbond", "salt", "pi"):
            label += " (Å)" if kind != "pi" else ""
        w = 0.34 + 0.12 + _text_width(label, 9.2) + 0.28
        entries.append((kind, label, w))
    avail = max((x1 - x0) - 0.6, 2.0)
    rows: list[list[tuple]] = []
    current: list[tuple] = []
    current_w = 0.0
    for entry in entries:
        if current and current_w + entry[2] > avail:
            rows.append(current)
            current = [entry]
            current_w = entry[2]
        else:
            current.append(entry)
            current_w += entry[2]
    if current:
        rows.append(current)
    for row_idx, row in enumerate(rows):
        total = sum(e[2] for e in row) - 0.28
        cursor = (x0 + x1) / 2.0 - total / 2.0
        yy = y + (len(rows) - 1 - row_idx) * 0.20
        for kind, label, w in row:
            color = COLORS[kind]
            if kind == "hydrophobic" and mode == "pl":
                ell = Ellipse((cursor + 0.17, yy), 0.24, 0.13, fill=False,
                              edgecolor=color, linewidth=0.8 * s, zorder=10)
                ax.add_patch(ell)
                for ang in np.linspace(0, 2 * pi, 8, endpoint=False):
                    ax.plot(
                        [cursor + 0.17 + 0.12 * cos(ang), cursor + 0.17 + 0.17 * cos(ang)],
                        [yy + 0.065 * sin(ang), yy + 0.095 * sin(ang)],
                        color=color, linewidth=0.7 * s, zorder=10,
                    )
            elif kind == "hydrophobic":
                ax.plot([cursor, cursor + 0.34], [yy, yy], color=color,
                        linewidth=0.9 * s, linestyle=(0, (1.0, 1.7)), zorder=10)
            elif kind == "pi":
                ax.plot([cursor, cursor + 0.34], [yy, yy], color=color,
                        linewidth=1.0 * s, linestyle=(0, (3, 2)), zorder=10)
            elif kind == "metal":
                ax.plot([cursor, cursor + 0.34], [yy, yy], color=color,
                        linewidth=1.0 * s, linestyle=(0, (2, 1.6)), zorder=10)
            else:
                ax.plot([cursor, cursor + 0.34], [yy, yy], color=color,
                        linewidth=1.0 * s, linestyle=(0, (3.5, 2.2)), zorder=10)
            ax.text(cursor + 0.46, yy, label, fontsize=fs, color="#333A40",
                    va="center", ha="left", family=FONT, zorder=10)
            cursor += w


def _setup_figure(width: float, height: float, scale: float, dpi: float) -> tuple[Figure, object]:
    fig = Figure(figsize=(max(width, 1e-3) * scale, max(height, 1e-3) * scale), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    ax.set_aspect("auto")
    ax.axis("off")
    return fig, ax


def _expand_bounds(x0, x1, y0, y1, min_width: float):
    if x1 - x0 < min_width:
        extra = (min_width - (x1 - x0)) / 2.0
        x0 -= extra
        x1 += extra
    return x0, x1, y0, y1


def _draw_title(ax, x0, x1, y1, title: str, subtitle: str, s: float):
    cx = (x0 + x1) / 2.0
    max_w = max((x1 - x0) - 0.6, 2.0)
    title_fs = 16.2
    while title_fs > 10.0 and _text_width(title, title_fs, "bold") > max_w:
        title_fs -= 0.3
    sub_fs = 9.5
    while sub_fs > 6.4 and _text_width(subtitle, sub_fs) > max_w:
        sub_fs -= 0.2
    ax.text(cx, y1 - 0.30, title, fontsize=title_fs * s, fontweight="bold",
            color=COLORS["text"], ha="center", va="top", family=FONT, zorder=20)
    ax.text(cx, y1 - 0.65, subtitle, fontsize=sub_fs * s, color=COLORS["subtext"],
            ha="center", va="top", family=FONT, zorder=20)


def _draw_footer(ax, x1, y0, s: float):
    ax.text(x1 - 0.18, y0 + 0.10, "Generated by PDBhx", fontsize=6.4 * s,
            color="#9AA4AE", ha="right", va="bottom", family=FONT)


# ---------------------------------------------------------------------------
# 蛋白-小分子示意图
# ---------------------------------------------------------------------------

def build_pl_figure(result: PLResult, scale: float = 1.0, dpi: float = 110,
                    show_distances: bool = True) -> Figure:
    lig = result.ligand
    chem = result.ligand_chem
    atoms = chem.atoms
    coords = np.array([a.coord for a in atoms])
    center, basis = _make_basis(coords)
    SCALE = 0.34
    uv = _project(coords, center, basis, SCALE)
    centroid2d = uv.mean(axis=0)

    # ---- 残基节点尺寸 ----
    node_info = []
    node_fs = 10.4
    for idx, node in enumerate(result.nodes):
        label = node.residue.label
        total_count = len(node.interactions)
        extra = 0.19 if total_count >= 2 else 0.0
        tw, th = text_extent_in(label, node_fs)
        w = tw + 0.24 + extra
        h = max(th + 0.16, 0.30)
        d3 = node.anchor - center
        d2 = basis @ d3
        if np.linalg.norm(d2[:2]) < 0.25:
            ang = (idx * 2.399963) % (2 * pi)
        else:
            ang = atan2(float(d2[1]), float(d2[0]))
        node_info.append({"node": node, "w": w, "h": h, "angle": ang, "pos": None,
                          "extra": extra, "total": total_count})

    # ---- 按角度排序并分配角度间隔 ----
    n = len(node_info)
    if n > 0:
        node_info.sort(key=lambda d: d["angle"])
        R_assume = max(2.2, 1.0 + 0.055 * n)
        needed = []
        for i in range(n):
            a = node_info[i]["w"]
            b = node_info[(i + 1) % n]["w"]
            needed.append((a + b) / 2.0 / R_assume + 0.055)
        total_needed = sum(needed)
        cap = 2 * pi * 0.93
        if total_needed <= cap:
            slack = cap - total_needed
            gaps = [x + slack / n for x in needed]
        else:
            factor = cap / total_needed
            gaps = [x * factor for x in needed]
        angle = node_info[0]["angle"]
        for i in range(n):
            node_info[i]["angle"] = angle
            angle += gaps[i]
        # 重新按角度排序（保持顺序，仅规范化）
        node_info.sort(key=lambda d: d["angle"])

    # 每个方向的配体外缘
    def lig_extent(theta: float) -> float:
        direction = np.array([cos(theta), sin(theta)])
        if len(uv) == 0:
            return 0.0
        proj = (uv - centroid2d) @ direction
        return float(np.max(proj))

    # 初始半径（仅疏水的残基更贴近配体，形成“睫毛”效果）
    for info in node_info:
        theta = info["angle"]
        half_support = 0.5 * (info["w"] * abs(cos(theta)) + info["h"] * abs(sin(theta)))
        extent = lig_extent(theta)
        only_hydro = info["node"].kinds == {"hydrophobic"}
        r_min = 1.62 if only_hydro else 1.75
        gap = 0.36 if only_hydro else 0.46
        info["r"] = max(r_min, extent + gap + half_support)
        info["pos"] = np.array([
            centroid2d[0] + info["r"] * cos(theta),
            centroid2d[1] + info["r"] * sin(theta),
        ])

    # ---- 标签防重叠 ----
    if n > 1:
        for _ in range(160):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = node_info[i], node_info[j]
                    dx = b["pos"][0] - a["pos"][0]
                    dy = b["pos"][1] - a["pos"][1]
                    ox = (a["w"] + b["w"]) / 2.0 + 0.10 - abs(dx)
                    oy = (a["h"] + b["h"]) / 2.0 + 0.06 - abs(dy)
                    if ox > 0 and oy > 0:
                        if ox <= oy:
                            shift = ox / 2.0 + 0.004
                            sign = 1.0 if dx >= 0 else -1.0
                            a["pos"][0] -= shift * sign
                            b["pos"][0] += shift * sign
                        else:
                            shift = oy / 2.0 + 0.004
                            sign = 1.0 if dy >= 0 else -1.0
                            a["pos"][1] -= shift * sign
                            b["pos"][1] += shift * sign
                        moved = True
            # 推离配体
            for info in node_info:
                vec = info["pos"] - centroid2d
                dist = float(np.linalg.norm(vec))
                theta = atan2(vec[1], vec[0])
                half_support = 0.5 * (info["w"] * abs(cos(theta)) + info["h"] * abs(sin(theta)))
                gap = 0.26 if info["node"].kinds == {"hydrophobic"} else 0.32
                min_r = lig_extent(theta) + gap + half_support
                if dist < min_r and dist > 1e-6:
                    info["pos"] = centroid2d + vec / dist * min_r
                    moved = True
            if not moved:
                break

    # ---- 边界与画布 ----
    xs = [uv[:, 0].min() - 0.2, uv[:, 0].max() + 0.2]
    ys = [uv[:, 1].min() - 0.2, uv[:, 1].max() + 0.2]
    for info in node_info:
        xs += [info["pos"][0] - info["w"] / 2 - 0.18, info["pos"][0] + info["w"] / 2 + 0.18]
        ys += [info["pos"][1] - info["h"] / 2 - 0.20, info["pos"][1] + info["h"] / 2 + 0.20]
    x0, x1 = min(xs) - 0.28, max(xs) + 0.28
    y0, y1 = min(ys) - 0.80, max(ys) + 0.95
    x0, x1, y0, y1 = _expand_bounds(x0, x1, y0, y1, 6.9)
    fig, ax = _setup_figure(x1 - x0, y1 - y0, scale, dpi)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # ---- 配体骨架 ----
    interacting_names = {
        it.atom_b for it in result.interactions
        if it.atom_b and it.atom_b != "ring"
    }
    _draw_ligand(ax, atoms, chem, uv, centroid2d, scale, interacting_names)
    if not node_info:
        ax.text(centroid2d[0], float(uv[:, 1].min()) - 0.52,
                "No interactions detected with the current thresholds",
                fontsize=9.5 * scale, color=COLORS["subtext"], ha="center", va="center",
                family=FONT, zorder=8)

    # ---- 连接线（氢键/盐桥/π）----
    coords2d_by_index = {i: uv[i] for i in range(len(atoms))}
    name_to_index: dict[str, int] = {}
    for i, atom in enumerate(atoms):
        name_to_index.setdefault(atom.name, i)

    counts = {k: 0 for k in ("hbond", "salt", "pi", "metal", "hydrophobic")}
    for it in result.interactions:
        counts[it.kind] += 1

    for info in node_info:
        node = info["node"]
        specific = [i for i in node.interactions
                    if i.kind in ("hbond", "salt", "pi", "metal")]
        specific.sort(key=lambda i: ("hbond", "salt", "pi", "metal").index(i.kind))
        drawn = 0
        for it in specific:
            if drawn >= 4:
                break
            if it.kind == "pi":
                target = _project(it.coord_b.reshape(1, 3), center, basis, SCALE)[0]
            else:
                idx = name_to_index.get(it.atom_b or "")
                if idx is None:
                    target = _project(it.coord_b.reshape(1, 3), center, basis, SCALE)[0]
                else:
                    target = coords2d_by_index[idx]
            ex, ey = _rect_edge(info["pos"][0], info["pos"][1], info["w"], info["h"],
                                target[0], target[1])
            color = COLORS[it.kind]
            ax.plot([ex, target[0]], [ey, target[1]], color=color,
                    linewidth=1.1 * scale, linestyle=(0, (3.6, 2.2)),
                    solid_capstyle="round", zorder=2)
            # 蛋白原子名
            dx, dy = target[0] - ex, target[1] - ey
            length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            ux, uy = dx / length, dy / length
            if show_distances:
                d_node = min(0.30 + 0.12 * drawn, 0.75 * length)
                lx = ex + ux * d_node - uy * (0.09 + 0.03 * drawn)
                ly = ey + uy * d_node + ux * (0.09 + 0.03 * drawn)
                label = "π" if it.kind == "pi" else f"{it.distance:.2f}"
                ax.text(lx, ly, label, fontsize=8.4 * scale, color=color,
                        ha="center", va="center", family=FONT, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.92))
            drawn += 1

    # ---- 残基标签 ----
    for info in node_info:
        node = info["node"]
        cx, cy = info["pos"]
        w, h = info["w"], info["h"]
        kinds = node.kinds
        if kinds == {"hydrophobic"}:
            hyd = max((it for it in node.interactions if it.kind == "hydrophobic"),
                      key=lambda it: it.contacts, default=None)
            if hyd is not None:
                target = _project(hyd.coord_b.reshape(1, 3), center, basis, SCALE)[0]
                ex, ey = _rect_edge(cx, cy, w, h, target[0], target[1])
                ax.plot([ex, target[0]], [ey, target[1]], color=COLORS["hydrophobic"],
                        linewidth=0.6 * scale, linestyle=(0, (1.2, 2.2)), alpha=0.6,
                        zorder=1.8)
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2.0, cy - h / 2.0), w, h,
            boxstyle=f"round,pad=0,rounding_size={0.06}",
            facecolor="white", edgecolor="none", zorder=3,
        ))
        if "hydrophobic" in kinds:
            _draw_spokes(ax, cx, cy, w, h, scale)
        if "hbond" in kinds:
            color = COLORS["hbond"]
        elif "salt" in kinds:
            color = COLORS["salt"]
        elif "pi" in kinds:
            color = COLORS["pi"]
        elif "metal" in kinds:
            color = COLORS["metal"]
        else:
            color = "#4A221C"
        extra = info.get("extra", 0.0)
        total_count = info.get("total", 0)
        ax.text(cx - extra / 2.0, cy, node.residue.label,
                fontsize=node_fs * scale, color=color,
                ha="center", va="center", family=FONT, zorder=6)
        if total_count >= 2:
            ax.text(cx + w / 2.0 - 0.08, cy, f"×{total_count}",
                    fontsize=6.8 * scale, color="#5E6B63",
                    ha="right", va="center", fontweight="bold",
                    family=FONT, zorder=7)
    # ---- 标题与图例 ----
    title = f"Ligand–Protein Interaction Diagram — {lig.resname} {lig.resseq}" \
            f"{lig.icode.strip()} (Chain {lig.chain.strip() or '-'})"
    subtitle_parts = [
        f"PDB {result.structure.pdb_id}",
        f"{len(result.nodes)} contacting residues",
        f"H-bonds {counts['hbond']}",
        f"Salt bridges {counts['salt']}",
        f"π–π {counts['pi']}",
    ]
    if counts["metal"]:
        subtitle_parts.append(f"Metal coordination {counts['metal']}")
    subtitle_parts.append(f"Hydrophobic {counts['hydrophobic']}")
    subtitle = "   ·   ".join(subtitle_parts)
    _draw_title(ax, x0, x1, y1, title, subtitle, scale)
    _draw_legend(ax, x0, x1, y0 + 0.34, scale,
                 show_pi=result.params.pi_enabled and counts["pi"] > 0,
                 mode="pl", show_metal=counts["metal"] > 0)
    _draw_footer(ax, x1, y0, scale)
    return fig


def _draw_ligand(ax, atoms, chem, uv: np.ndarray, centroid2d: np.ndarray, s: float,
                 interacting_names: set[str] | None = None):
    ring_bonds: set[tuple[int, int]] = set()
    for ring in chem.rings:
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            ring_bonds.add((min(a, b), max(a, b)))

    # 羧酸/羧酸根的两个 C-O 键按离域处理，不画双键
    carboxylate_carbons: set[int] = set()
    for i, atom in enumerate(atoms):
        if atom.element.upper() != "C":
            continue
        oxy = [j for j in chem.neighbors.get(i, [])
               if atoms[j].element.upper() == "O"
               and float(np.linalg.norm(atom.coord - atoms[j].coord)) < 1.36]
        if len(oxy) >= 2:
            carboxylate_carbons.add(i)

    # 酰胺 C-N 具有部分双键性质，不按双键绘制
    amide_bonds: set[tuple[int, int]] = set()
    for i, atom in enumerate(atoms):
        if atom.element.upper() != "C":
            continue
        nbrs = chem.neighbors.get(i, [])
        has_carbonyl = any(
            atoms[j].element.upper() == "O"
            and float(np.linalg.norm(atom.coord - atoms[j].coord)) < 1.34
            for j in nbrs
        )
        if not has_carbonyl:
            continue
        for j in nbrs:
            if atoms[j].element.upper() == "N":
                amide_bonds.add((min(i, j), max(i, j)))

    double_threshold = {
        ("C", "C"): 1.42, ("C", "N"): 1.36, ("C", "O"): 1.33,
        ("C", "S"): 1.72, ("N", "N"): 1.30, ("N", "O"): 1.32,
        ("O", "P"): 1.60, ("O", "S"): 1.52,
    }
    double_bonds: set[tuple[int, int]] = set()
    for i, j in chem.bonds:
        key = (min(i, j), max(i, j))
        if key in ring_bonds or key in amide_bonds:
            continue
        if i in carboxylate_carbons or j in carboxylate_carbons:
            continue
        pair = tuple(sorted((atoms[i].element.upper(), atoms[j].element.upper())))
        threshold = double_threshold.get(pair)
        if threshold is None:
            continue
        if float(np.linalg.norm(atoms[i].coord - atoms[j].coord)) < threshold:
            double_bonds.add(key)

    # 芳香环虚线圆
    for ring in chem.rings:
        coords = np.array([atoms[i].coord for i in ring])
        mean_len = np.mean([np.linalg.norm(atoms[ring[k]].coord - atoms[ring[(k + 1) % len(ring)]].coord)
                            for k in range(len(ring))])
        center3 = coords.mean(axis=0)
        center2 = uv[ring].mean(axis=0)
        radius = float(np.mean(np.linalg.norm(uv[ring] - center2, axis=1)))
        if mean_len < 1.46 and radius > 0.12:
            circ = Circle(tuple(center2), radius * 0.62, fill=False, edgecolor="#8A9299",
                          linewidth=0.45 * s, linestyle=(0, (1.0, 1.6)), zorder=3.5)
            ax.add_patch(circ)

    for i, j in chem.bonds:
        p1, p2 = uv[i], uv[j]
        mid = (p1 + p2) / 2.0
        c1 = element_color(atoms[i].element)
        c2 = element_color(atoms[j].element)
        ax.plot([p1[0], mid[0]], [p1[1], mid[1]], color=c1, linewidth=1.85 * s,
                solid_capstyle="round", zorder=4)
        ax.plot([mid[0], p2[0]], [mid[1], p2[1]], color=c2, linewidth=1.85 * s,
                solid_capstyle="round", zorder=4)
        if (min(i, j), max(i, j)) in double_bonds:
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = max((dx * dx + dy * dy) ** 0.5, 1e-9)
            ox, oy = -dy / length * 0.048, dx / length * 0.048
            # 双键线放在远离分子中心一侧，减少与其它键重叠
            rel = ((p1[0] + p2[0]) / 2.0 - centroid2d[0],
                   (p1[1] + p2[1]) / 2.0 - centroid2d[1])
            if ox * rel[0] + oy * rel[1] > 0:
                ox, oy = -ox, -oy
            q1 = (p1[0] + ox, p1[1] + oy)
            q2 = (p2[0] + ox, p2[1] + oy)
            qmid = ((q1[0] + q2[0]) / 2.0, (q1[1] + q2[1]) / 2.0)
            ax.plot([q1[0], qmid[0]], [q1[1], qmid[1]], color=c1, linewidth=1.05 * s,
                    solid_capstyle="round", zorder=4)
            ax.plot([qmid[0], q2[0]], [qmid[1], q2[1]], color=c2, linewidth=1.05 * s,
                    solid_capstyle="round", zorder=4)

    for i, atom in enumerate(atoms):
        p = uv[i]
        r = element_radius(atom.element) * (1.0 if atom.element.upper() != "C" else 0.8) * 1.05
        ax.add_patch(Circle(tuple(p), r, facecolor=element_color(atom.element),
                            edgecolor="white", linewidth=0.3 * s, zorder=5))

    # 杂原子标签，向远离质心方向排布并做简单避让
    label_items = []
    for i, atom in enumerate(atoms):
        if atom.element.upper() == "C":
            continue
        if interacting_names is not None and atom.name not in interacting_names:
            continue
        p = uv[i]
        vec = p - centroid2d
        norm = float(np.linalg.norm(vec))
        direction = vec / norm if norm > 1e-6 else np.array([0.0, 1.0])
        lp = p + direction * 0.17
        tw, th = text_extent_in(atom.name, 5.8)
        label_items.append({
            "pos": np.array(lp, dtype=float), "anchor": np.array(lp, dtype=float),
            "w": tw + 0.05, "h": th + 0.04,
            "name": atom.name, "color": element_color(atom.element),
        })
    if len(label_items) > 1:
        for _ in range(220):
            moved = False
            for a in range(len(label_items)):
                for b in range(a + 1, len(label_items)):
                    it1, it2 = label_items[a], label_items[b]
                    dx = it2["pos"][0] - it1["pos"][0]
                    dy = it2["pos"][1] - it1["pos"][1]
                    ox = (it1["w"] + it2["w"]) / 2.0 + 0.015 - abs(dx)
                    oy = (it1["h"] + it2["h"]) / 2.0 + 0.015 - abs(dy)
                    if ox > 0 and oy > 0:
                        if ox <= oy:
                            shift = ox / 2.0 + 0.004
                            sign = 1.0 if dx >= 0 else -1.0
                            it1["pos"][0] -= shift * sign
                            it2["pos"][0] += shift * sign
                        else:
                            shift = oy / 2.0 + 0.004
                            sign = 1.0 if dy >= 0 else -1.0
                            it1["pos"][1] -= shift * sign
                            it2["pos"][1] += shift * sign
                        moved = True
            for item in label_items:
                item["pos"] += (item["anchor"] - item["pos"]) * 0.012
                for p in uv:
                    delta = item["pos"] - p
                    dist = float(np.linalg.norm(delta))
                    if 1e-9 < dist < 0.095:
                        item["pos"] = p + delta / dist * 0.095
                        moved = True
            if not moved:
                break

    for item in label_items:
        ax.text(item["pos"][0], item["pos"][1], item["name"], fontsize=8.0 * s,
                color=item["color"], ha="center", va="center",
                family=FONT, zorder=6,
                bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.75))


def _draw_spokes(ax, cx, cy, w, h, s):
    color = COLORS["spoke"]
    ell = Ellipse((cx, cy), w + 0.20, h + 0.15, fill=False, edgecolor=color,
                  linewidth=0.75 * s, zorder=3.2)
    ax.add_patch(ell)
    for ang in np.linspace(0, 2 * pi, 14, endpoint=False):
        x1 = cx + (w / 2.0 + 0.09) * cos(ang)
        y1 = cy + (h / 2.0 + 0.065) * sin(ang)
        x2 = cx + (w / 2.0 + 0.16) * cos(ang)
        y2 = cy + (h / 2.0 + 0.125) * sin(ang)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.65 * s,
                solid_capstyle="round", zorder=3.2)

# ---------------------------------------------------------------------------
# 蛋白-蛋白界面图
# ---------------------------------------------------------------------------

def build_pp_figure(result: PPResult, scale: float = 1.0, dpi: float = 110,
                    show_distances: bool = True) -> Figure:
    nodes_a = result.nodes_a
    nodes_b = result.nodes_b
    if not nodes_a and not nodes_b:
        return _build_empty_pp(result, scale, dpi)

    # ---- 垂直顺序：基于 3D 接触方向，再做重心迭代 ----
    _order_pp_nodes(result, nodes_a, nodes_b)

    dy = 0.54 if max(len(nodes_a), len(nodes_b)) <= 24 else 0.48
    node_fs = 10.4 if max(len(nodes_a), len(nodes_b)) <= 30 else 9.2
    info_map: dict[tuple, dict] = {}
    max_w_a = 0.0
    max_w_b = 0.0
    for side, nodes in (("A", nodes_a), ("B", nodes_b)):
        count = len(nodes)
        for idx, node in enumerate(nodes):
            label = node.residue.label
            total_count = len(node.interactions)
            extra = 0.19 if total_count >= 2 else 0.0
            tw, th = text_extent_in(label, node_fs)
            w = tw + 0.24 + extra
            h = max(th + 0.16, 0.30)
            y = (count - 1) / 2.0 * dy - idx * dy
            info_map[(side, node.residue.key)] = {
                "node": node, "w": w, "h": h, "y": y, "x": 0.0,
                "extra": extra, "total": total_count,
            }
            if side == "A":
                max_w_a = max(max_w_a, w)
            else:
                max_w_b = max(max_w_b, w)

    gap = 1.90
    for side, nodes in (("A", nodes_a), ("B", nodes_b)):
        for node in nodes:
            info = info_map[(side, node.residue.key)]
            if side == "A":
                info["x"] = -(gap / 2.0 + info["w"] / 2.0)
            else:
                info["x"] = +(gap / 2.0 + info["w"] / 2.0)

    # ---- 边界 ----
    xs = [-gap / 2 - 0.35, gap / 2 + 0.35]
    ys = [0.0]
    for info in info_map.values():
        xs += [info["x"] - info["w"] / 2 - 0.12, info["x"] + info["w"] / 2 + 0.12]
        ys += [info["y"] - info["h"] / 2 - 0.12, info["y"] + info["h"] / 2 + 0.12]
    y_min_nodes = min(ys)
    y_max_nodes = max(ys)
    x0, x1 = min(xs) - 0.45, max(xs) + 0.45
    y0 = y_min_nodes - 0.70
    y1 = y_max_nodes + 1.20
    x0, x1, y0, y1 = _expand_bounds(x0, x1, y0, y1, 6.8)
    fig, ax = _setup_figure(x1 - x0, y1 - y0, scale, dpi)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # ---- 链背景面板 ----
    _draw_chain_panel(ax, nodes_a, info_map, "A", scale, y_min_nodes, y_max_nodes)
    _draw_chain_panel(ax, nodes_b, info_map, "B", scale, y_min_nodes, y_max_nodes)

    # ---- 边 ----
    groups: dict[tuple, list[RawInteraction]] = {}
    for it in result.interactions:
        if it.res_b is None:
            continue
        groups.setdefault((it.res_a.key, it.res_b.key), []).append(it)

    priority = {"hbond": 0, "salt": 1, "pi": 2, "hydrophobic": 3}
    edge_specs = []
    for (ka, kb), items in groups.items():
        ia = info_map.get(("A", ka))
        ib = info_map.get(("B", kb))
        if ia is None or ib is None:
            continue
        kinds = {it.kind for it in items}
        primary = sorted(kinds, key=lambda k: priority[k])
        for kind in primary:
            subset = [it for it in items if it.kind == kind]
            if kind == "hydrophobic":
                dist = min(it.distance for it in subset)
                label = ""
            elif kind == "pi":
                dist = min(it.distance for it in subset)
                label = "π"
            else:
                dist = min(it.distance for it in subset)
                label = f"{dist:.2f}"
            edge_specs.append((ia, ib, kind, dist, label, len(subset)))

    order_z = {"hydrophobic": 1, "hbond": 3, "pi": 4, "salt": 5}
    edge_labels: list[dict] = []

    for ia, ib, kind, dist, label, count in edge_specs:
        x1e, y1e = _rect_edge(ia["x"], ia["y"], ia["w"], ia["h"], ib["x"], ib["y"])
        x2e, y2e = _rect_edge(ib["x"], ib["y"], ib["w"], ib["h"], ia["x"], ia["y"])
        color = COLORS[kind]
        rad = {"hydrophobic": 0.0, "hbond": 0.06, "pi": 0.11, "salt": -0.07}[kind]
        style = {
            "hydrophobic": (0, (1.0, 1.7)),
            "hbond": (0, (3.6, 2.2)),
            "salt": (0, (4.0, 2.0)),
            "pi": (0, (3.0, 2.0)),
        }[kind]
        lw = {"hydrophobic": 0.9, "hbond": 1.0, "pi": 1.0, "salt": 1.15}[kind]
        patch = FancyArrowPatch(
            (x1e, y1e), (x2e, y2e),
            connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
            color=color, linewidth=lw * scale, linestyle=style,
            shrinkA=0, shrinkB=0, zorder=order_z[kind],
            alpha=0.6 if kind == "hydrophobic" else 1.0,
        )
        ax.add_patch(patch)

        if label and show_distances:
            frac = 0.38
            mx = x1e + (x2e - x1e) * frac
            my = y1e + (y2e - y1e) * frac
            dx, dyv = x2e - x1e, y2e - y1e
            length = max((dx * dx + dyv * dyv) ** 0.5, 1e-6)
            nx, ny = -dyv / length, dx / length
            offset = 0.065 * np.sign(rad if rad != 0 else 1.0)
            lx, ly = mx + nx * offset, my + ny * offset
            tw, th = text_extent_in(label, 5.0)
            edge_labels.append({
                "pos": np.array([lx, ly], dtype=float), "w": tw + 0.05,
                "h": max(th + 0.03, 0.08), "label": label, "color": color,
            })

    _relax_labels(edge_labels, iterations=140, margin=0.012)
    for item in edge_labels:
        ax.text(item["pos"][0], item["pos"][1], item["label"], fontsize=6.0 * scale,
                color=item["color"], ha="center", va="center", family=FONT, zorder=5,
                bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="none", alpha=0.92))

    # ---- 节点 ----
    for side, nodes in (("A", nodes_a), ("B", nodes_b)):
        for node in nodes:
            info = info_map[(side, node.residue.key)]
            cx, cy, w, h = info["x"], info["y"], info["w"], info["h"]
            edge_color = COLORS["chain_a"] if side == "A" else COLORS["chain_b"]
            fill = "#FFFFFF"
            ax.add_patch(FancyBboxPatch(
                (cx - w / 2.0, cy - h / 2.0), w, h,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=fill, edgecolor=edge_color, linewidth=0.8 * scale, zorder=6,
            ))
            extra = info.get("extra", 0.0)
            total_count = info.get("total", 0)
            ax.text(cx - extra / 2.0, cy, node.residue.label, fontsize=node_fs * scale,
                    color="#22303A", ha="center", va="center", family=FONT, zorder=7)
            if total_count >= 2:
                ax.text(cx + w / 2.0 - 0.08, cy, f"×{total_count}",
                        fontsize=6.8 * scale, color="#5E6B63", ha="right",
                        va="center", fontweight="bold", family=FONT, zorder=8)

    # ---- 链标题 ----
    _draw_chain_header(ax, info_map, nodes_a, "A", scale, y_max_nodes)
    _draw_chain_header(ax, info_map, nodes_b, "B", scale, y_max_nodes)

    # ---- 标题与图例 ----
    ca = result.chain_a.strip() or "-"
    cb = result.chain_b.strip() or "-"
    title = f"Protein–Protein Interaction Interface — Chain {ca} / Chain {cb}"
    subtitle = (
        f"PDB {result.structure.pdb_id}   ·   Interface residues {len(result.interface_a)} + "
        f"{len(result.interface_b)}   ·   H-bonds {result.stats['hbond']}   ·   "
        f"Salt bridges {result.stats['salt']}   ·   π–π {result.stats['pi']}   ·   "
        f"Hydrophobic {result.stats['hydrophobic']}"
    )
    _draw_title(ax, x0, x1, y1, title, subtitle, scale)
    _draw_legend(ax, x0, x1, y0 + 0.34, scale,
                 show_pi=result.params.pi_enabled and result.stats["pi"] > 0, mode="pp")
    _draw_footer(ax, x1, y0, scale)
    return fig


def _order_pp_nodes(result: PPResult, nodes_a, nodes_b):
    all_nodes = list(nodes_a) + list(nodes_b)
    if not all_nodes:
        return
    anchors = np.array([n.anchor for n in all_nodes])
    mean_a = np.mean([n.anchor for n in nodes_a], axis=0) if nodes_a else anchors.mean(axis=0)
    mean_b = np.mean([n.anchor for n in nodes_b], axis=0) if nodes_b else anchors.mean(axis=0)
    axis = mean_a - mean_b
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / norm
    candidates = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    vertical = min(candidates, key=lambda c: abs(float(np.dot(c, axis))))
    vertical = vertical - np.dot(vertical, axis) * axis
    if float(np.linalg.norm(vertical)) < 1e-6:
        vertical = np.array([0.0, 0.0, 1.0])
    vertical = vertical / np.linalg.norm(vertical)

    def initial_sorted(nodes):
        return sorted(nodes, key=lambda n: float(np.dot(n.anchor, vertical)))

    nodes_a[:] = initial_sorted(nodes_a)
    nodes_b[:] = initial_sorted(nodes_b)

    partners: dict[tuple, list[tuple]] = {}
    for it in result.interactions:
        if it.res_b is None:
            continue
        partners.setdefault(("A", it.res_a.key), []).append(("B", it.res_b.key))
        partners.setdefault(("B", it.res_b.key), []).append(("A", it.res_a.key))

    def y_map(side, nodes):
        count = len(nodes)
        return {n.residue.key: (count - 1) / 2.0 - idx for idx, n in enumerate(nodes)}

    for _ in range(6):
        for side, nodes in (("A", nodes_a), ("B", nodes_b)):
            other_side = "B" if side == "A" else "A"
            other_y = y_map(other_side, nodes_b if other_side == "B" else nodes_a)
            current_y = y_map(side, nodes)

            def barycenter(node):
                linked = partners.get((side, node.residue.key), [])
                vals = [other_y[k] for s, k in linked if k in other_y]
                if not vals:
                    return current_y[node.residue.key]
                return sum(vals) / len(vals)

            nodes.sort(key=barycenter, reverse=True)


def _draw_chain_panel(ax, nodes, info_map, side, s, y_min, y_max):
    if not nodes:
        return
    infos = [info_map[(side, n.residue.key)] for n in nodes]
    x_left = min(i["x"] - i["w"] / 2 for i in infos) - 0.22
    x_right = max(i["x"] + i["w"] / 2 for i in infos) + 0.22
    y_bottom = min(i["y"] - i["h"] / 2 for i in infos) - 0.42
    y_top = max(i["y"] + i["h"] / 2 for i in infos) + 0.72
    fill = COLORS["panel_a"] if side == "A" else COLORS["panel_b"]
    ax.add_patch(FancyBboxPatch(
        (x_left, y_bottom), x_right - x_left, y_top - y_bottom,
        boxstyle="round,pad=0,rounding_size=0.14",
        facecolor=fill, edgecolor="none", zorder=0,
    ))


def _draw_chain_header(ax, info_map, nodes, side, s, y_max):
    if not nodes:
        return
    infos = [info_map[(side, n.residue.key)] for n in nodes]
    top = max(i["y"] + i["h"] / 2 for i in infos)
    cx = sum(i["x"] for i in infos) / len(infos)
    color = COLORS["chain_a"] if side == "A" else COLORS["chain_b"]
    chain_name = nodes[0].residue.chain.strip() or "-"
    ax.text(cx, top + 0.42, f"Chain {chain_name}", fontsize=12.4 * s,
            fontweight="bold", color=color, ha="center", va="center",
            family=FONT, zorder=9)


def _build_empty_pp(result: PPResult, scale: float, dpi: float) -> Figure:
    fig, ax = _setup_figure(7.6, 4.6, scale, dpi)
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 4.6)
    ca = result.chain_a.strip() or "-"
    cb = result.chain_b.strip() or "-"
    ax.text(3.8, 3.0, f"Chain {ca}  —  Chain {cb}", fontsize=16 * scale,
            ha="center", va="center", color=COLORS["text"], family=FONT)
    ax.text(3.8, 2.2, "No interfacial interactions detected with the current thresholds.",
            fontsize=8.5 * scale, ha="center", va="center", color=COLORS["subtext"], family=FONT)
    ax.text(3.8, 1.85, "Try relaxing distance / angle cutoffs and analyze again.",
            fontsize=8.5 * scale, ha="center", va="center", color=COLORS["subtext"], family=FONT)
    ax.text(3.8, 3.9, "Protein–Protein Interface", fontsize=13 * scale, fontweight="bold",
            ha="center", va="center", color=COLORS["text"], family=FONT)
    return fig
