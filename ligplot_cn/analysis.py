"""把相互作用检测结果组织为绘图与表格所需的模型。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .interactions import (
    InteractionParams,
    RawInteraction,
    analyze_ligand,
    detect_pl_hbonds,
    detect_pl_hydrophobic,
    detect_pl_metal,
    detect_pl_pi,
    detect_pl_salt,
    detect_pp_hbonds,
    detect_pp_hydrophobic,
    detect_pp_pi,
    detect_pp_salt,
    grid_neighbors,
    merge_hbond_salt,
    protein_interface_residues,
)
from .pdbio import Atom, Residue, Structure

KIND_ORDER = {"hbond": 0, "salt": 1, "pi": 2, "metal": 3, "hydrophobic": 4}
KIND_NAMES_CN = {
    "hbond": "氢键",
    "salt": "盐桥",
    "pi": "π–π 堆积",
    "metal": "配位键",
    "hydrophobic": "疏水接触",
}
KIND_NAMES_EN = {
    "hbond": "Hydrogen bond",
    "salt": "Salt bridge",
    "pi": "π–π stacking",
    "metal": "Metal coordination",
    "hydrophobic": "Hydrophobic contact",
}

BACKBONE_POLAR = {"N", "O", "OXT", "OT1", "OT2"}


def _hbond_nature(it: RawInteraction, mode: str) -> str:
    a_backbone = (it.atom_a or "") in BACKBONE_POLAR
    if mode == "pl":
        return "主链氢键" if a_backbone else "侧链氢键"
    b_backbone = (it.atom_b or "") in BACKBONE_POLAR
    if a_backbone and b_backbone:
        return "主链–主链氢键"
    if a_backbone or b_backbone:
        return "主链–侧链氢键"
    return "侧链–侧链氢键"


def interaction_nature(it: RawInteraction, mode: str) -> str:
    if it.kind == "hbond":
        return _hbond_nature(it, mode)
    if it.kind == "salt":
        return "盐桥（静电）"
    if it.kind == "pi":
        return f"{it.note} π–π" if it.note else "π–π 堆积"
    if it.kind == "metal":
        return "金属配位"
    return "疏水接触"


def _residue_statistics(nodes, mode: str) -> tuple[list[dict], list[tuple]]:
    stats: list[dict] = []
    for node in nodes:
        per = {"hbond": 0, "salt": 0, "pi": 0, "metal": 0, "hydrophobic": 0}
        for it in node.interactions:
            per[it.kind] += 1
        total = sum(per.values())
        min_dist = min((it.distance for it in node.interactions), default=0.0)
        stats.append({
            "label": node.residue.label_spaced,
            "key": node.residue.key,
            "hbond": per["hbond"], "salt": per["salt"], "pi": per["pi"],
            "metal": per["metal"], "hydrophobic": per["hydrophobic"],
            "total": total, "min_dist": min_dist,
            "contacts": sum(it.contacts for it in node.interactions
                            if it.kind == "hydrophobic"),
        })
    stats.sort(key=lambda s: (-s["total"], s["min_dist"], s["label"]))
    rows = [
        (
            s["label"], s["hbond"], s["salt"], s["pi"], s["metal"],
            s["hydrophobic"], s["total"], f"{s['min_dist']:.2f}",
        )
        for s in stats
    ]
    return stats, rows


@dataclass
class PLNode:
    residue: Residue
    interactions: list[RawInteraction]
    anchor: np.ndarray
    kinds: set[str] = field(default_factory=set)


@dataclass
class PLResult:
    structure: Structure
    ligand: Residue
    nodes: list[PLNode]
    interactions: list[RawInteraction]
    table_rows: list[tuple]
    residue_rows: list[tuple]
    residue_stats: list[dict]
    summary: str
    stats: dict
    ligand_chem: object
    params: InteractionParams

    @property
    def mode(self) -> str:
        return "pl"


@dataclass
class PPNode:
    residue: Residue
    side: str
    interactions: list[RawInteraction]
    anchor: np.ndarray
    kinds: set[str] = field(default_factory=set)


@dataclass
class PPResult:
    structure: Structure
    chain_a: str
    chain_b: str
    nodes_a: list[PPNode]
    nodes_b: list[PPNode]
    interactions: list[RawInteraction]
    interface_a: list[Residue]
    interface_b: list[Residue]
    table_rows: list[tuple]
    residue_rows: list[tuple]
    residue_stats: list[dict]
    summary: str
    stats: dict
    params: InteractionParams

    @property
    def mode(self) -> str:
        return "pp"


# ---------------------------------------------------------------------------
# 公共
# ---------------------------------------------------------------------------

def _counts(interactions: list[RawInteraction]) -> dict[str, int]:
    counts = {"hbond": 0, "salt": 0, "pi": 0, "metal": 0, "hydrophobic": 0}
    for it in interactions:
        counts[it.kind] += 1
    return counts


def _ligand_name(lig: Residue) -> str:
    chain = lig.chain.strip() or "-"
    return f"{lig.resname} {lig.resseq}{lig.icode.strip()} (链 {chain})"


def _sort_key(it: RawInteraction):
    return (
        KIND_ORDER.get(it.kind, 9),
        it.res_a.chain,
        it.res_a.resseq,
        it.res_a.resname,
        it.atom_a or "",
    )


def _aggregate_salts(interactions: list[RawInteraction]) -> list[RawInteraction]:
    """同一残基对之间的多个带电原子对合并为一条盐桥（保留最短距离）。"""
    out: list[RawInteraction] = []
    index: dict[tuple, RawInteraction] = {}
    for it in interactions:
        if it.kind != "salt":
            out.append(it)
            continue
        key = (it.res_a.key, it.res_b.key if it.res_b else None)
        prev = index.get(key)
        if prev is None:
            it.contacts = 1
            index[key] = it
            out.append(it)
        else:
            prev.contacts += 1
            if it.distance < prev.distance:
                prev.atom_a, prev.coord_a = it.atom_a, it.coord_a
                prev.atom_b, prev.coord_b = it.atom_b, it.coord_b
                prev.distance = it.distance
    return out


def _residues_near_ligand(
    structure: Structure, ligand: Residue, cutoff: float
) -> list[Residue]:
    lig_coords = np.array([a.coord for a in ligand.atoms if a.is_heavy])
    if len(lig_coords) == 0:
        return []
    atoms: list[Atom] = []
    owners: list[Residue] = []
    for res in structure.residues:
        if res.is_water or res is ligand or res.key == ligand.key:
            continue
        if res.is_hetero and not res.is_amino:
            continue
        for atom in res.atoms:
            if atom.is_heavy:
                atoms.append(atom)
                owners.append(res)
    if not atoms:
        return []
    coords = np.array([a.coord for a in atoms])
    ia, _, _ = grid_neighbors(coords, lig_coords, cutoff)
    seen: set[tuple] = set()
    out: list[Residue] = []
    for idx in ia:
        res = owners[int(idx)]
        if res.key in seen:
            continue
        seen.add(res.key)
        out.append(res)
    order = {r.key: k for k, r in enumerate(structure.residues)}
    out.sort(key=lambda r: order.get(r.key, 0))
    return out


# ---------------------------------------------------------------------------
# 蛋白-小分子
# ---------------------------------------------------------------------------

def analyze_pl(
    structure: Structure, ligand: Residue, params: InteractionParams
) -> PLResult:
    chem = analyze_ligand(ligand)
    max_cut = max(
        params.hbond_dist, params.hydrophobic_dist,
        params.salt_dist, params.pi_dist if params.pi_enabled else 0.0,
        5.0,
    ) + 1.0
    residues = _residues_near_ligand(structure, ligand, max_cut)

    interactions: list[RawInteraction] = []
    interactions += detect_pl_hbonds(residues, ligand, chem, params)
    interactions += detect_pl_salt(residues, ligand, chem, params)
    interactions += detect_pl_metal(residues, ligand, chem, params)
    if params.pi_enabled:
        interactions += detect_pl_pi(residues, ligand, chem, params)
    interactions += detect_pl_hydrophobic(residues, ligand, chem, params)
    interactions = merge_hbond_salt(interactions)
    interactions = _aggregate_salts(interactions)
    interactions.sort(key=_sort_key)

    grouped: dict[tuple, list[RawInteraction]] = {}
    for it in interactions:
        grouped.setdefault(it.res_a.key, []).append(it)
    nodes: list[PLNode] = []
    for key, items in grouped.items():
        res = items[0].res_a
        anchor = np.mean([it.coord_a for it in items], axis=0)
        kinds = {it.kind for it in items}
        nodes.append(PLNode(residue=res, interactions=items, anchor=anchor, kinds=kinds))
    order = {r.key: k for k, r in enumerate(structure.residues)}
    nodes.sort(key=lambda n: order.get(n.residue.key, 0))

    rows: list[tuple] = []
    for it in interactions:
        angle = f"{it.angle:.0f}" if it.angle is not None else "—"
        contacts = str(it.contacts) if it.kind in ("hydrophobic", "salt", "metal") else "—"
        atom_a = "芳环" if it.atom_a == "ring" else (it.atom_a or "—")
        atom_b = "芳环" if it.atom_b == "ring" else (it.atom_b or "—")
        rows.append((
            KIND_NAMES_CN[it.kind],
            interaction_nature(it, "pl"),
            it.res_a.label_spaced,
            atom_a,
            atom_b,
            f"{it.distance:.2f}",
            angle,
            contacts,
        ))

    counts = _counts(interactions)
    residue_stats, residue_rows = _residue_statistics(nodes, "pl")
    stats = {
        **counts,
        "n_residues": len(nodes),
        "n_atoms": len(chem.atoms),
        "min_dist": min((it.distance for it in interactions), default=0.0),
    }
    summary = _pl_summary(structure, ligand, nodes, interactions, counts)
    return PLResult(
        structure=structure, ligand=ligand, nodes=nodes, interactions=interactions,
        table_rows=rows, residue_rows=residue_rows, residue_stats=residue_stats,
        summary=summary, stats=stats, ligand_chem=chem, params=params,
    )


def _pl_summary(
    structure: Structure,
    ligand: Residue,
    nodes: list[PLNode],
    interactions: list[RawInteraction],
    counts: dict[str, int],
) -> str:
    lines = [
        f"结构文件：{structure.pdb_id}",
        f"配体：{_ligand_name(ligand)}    重原子数：{len(ligand.heavy_atoms)}",
        f"结合口袋残基：{len(nodes)}",
        "",
        f"氢键：{counts['hbond']}    盐桥：{counts['salt']}    "
        f"π–π 堆积：{counts['pi']}    配位键：{counts['metal']}    "
        f"疏水接触：{counts['hydrophobic']}",
    ]
    if interactions:
        min_it = min(interactions, key=lambda x: x.distance)
        lines.append(f"最短接触：{min_it.distance:.2f} Å  "
                     f"（{min_it.res_a.label_spaced} · {min_it.atom_a or '疏水'}）")
    else:
        lines.append("未检测到符合阈值的相互作用，可适当放宽距离或角度阈值。")

    def res_list(kind: str) -> list[str]:
        seen = []
        for it in interactions:
            if it.kind == kind and it.res_a.label not in seen:
                seen.append(it.res_a.label)
        return seen

    for kind in ("hbond", "salt", "pi", "metal", "hydrophobic"):
        names = res_list(kind)
        if names:
            lines.append("")
            lines.append(f"{KIND_NAMES_CN[kind]}残基：")
            lines.append("  " + "、".join(names))

    counts_per_res: dict[str, int] = {}
    for it in interactions:
        counts_per_res[it.res_a.label] = counts_per_res.get(it.res_a.label, 0) + 1
    top = sorted(counts_per_res.items(), key=lambda kv: -kv[1])[:5]
    if top:
        lines.append("")
        lines.append("相互作用最多的残基：")
        for name, n in top:
            lines.append(f"  {name}：{n} 项")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 蛋白-蛋白
# ---------------------------------------------------------------------------

def analyze_pp(
    structure: Structure, chain_a: str, chain_b: str, params: InteractionParams
) -> PPResult:
    res_a_all = structure.chain_residues(chain_a, protein_only=True)
    res_b_all = structure.chain_residues(chain_b, protein_only=True)
    if not res_a_all or not res_b_all:
        raise ValueError("所选链中没有可分析的蛋白质残基。")

    interface_a, interface_b = protein_interface_residues(
        res_a_all, res_b_all, max(params.interface_dist, params.hydrophobic_dist)
    )

    interactions: list[RawInteraction] = []
    interactions += detect_pp_hbonds(interface_a, interface_b, params)
    interactions += detect_pp_salt(interface_a, interface_b, params)
    if params.pi_enabled:
        interactions += detect_pp_pi(interface_a, interface_b, params)
    interactions += detect_pp_hydrophobic(interface_a, interface_b, params)
    interactions = merge_hbond_salt(interactions)
    interactions = _aggregate_salts(interactions)
    interactions.sort(key=_sort_key)

    def build_nodes(side_residues: list[Residue], side: str) -> list[PPNode]:
        grouped: dict[tuple, list[RawInteraction]] = {}
        for it in interactions:
            if side == "A" and it.res_a.chain == chain_a:
                grouped.setdefault(it.res_a.key, []).append(it)
            elif side == "B" and it.res_b is not None and it.res_b.chain == chain_b:
                grouped.setdefault(it.res_b.key, []).append(it)
        nodes = []
        for key, items in grouped.items():
            if side == "A":
                res = items[0].res_a
                anchor = np.mean([it.coord_a for it in items], axis=0)
            else:
                res = items[0].res_b
                anchor = np.mean([it.coord_b for it in items], axis=0)
            nodes.append(PPNode(
                residue=res, side=side, interactions=items, anchor=anchor,
                kinds={it.kind for it in items},
            ))
        order = {r.key: k for k, r in enumerate(side_residues)}
        nodes.sort(key=lambda n: order.get(n.residue.key, 0))
        return nodes

    nodes_a = build_nodes(interface_a, "A")
    nodes_b = build_nodes(interface_b, "B")

    rows: list[tuple] = []
    for it in interactions:
        angle = f"{it.angle:.0f}" if it.angle is not None else "—"
        contacts = str(it.contacts) if it.kind in ("hydrophobic", "salt", "metal") else "—"
        atom_a = "芳环" if it.atom_a == "ring" else (it.atom_a or "—")
        atom_b = "芳环" if it.atom_b == "ring" else (it.atom_b or "—")
        rows.append((
            KIND_NAMES_CN[it.kind],
            interaction_nature(it, "pp"),
            it.res_a.label_spaced,
            atom_a,
            it.res_b.label_spaced if it.res_b else "—",
            atom_b,
            f"{it.distance:.2f}",
            angle,
            contacts,
        ))

    counts = _counts(interactions)
    residue_stats, residue_rows = _residue_statistics(nodes_a + nodes_b, "pp")
    stats = {
        **counts,
        "n_iface_a": len(interface_a),
        "n_iface_b": len(interface_b),
        "n_nodes_a": len(nodes_a),
        "n_nodes_b": len(nodes_b),
        "min_dist": min((it.distance for it in interactions), default=0.0),
    }
    summary = _pp_summary(structure, chain_a, chain_b, interface_a, interface_b,
                          nodes_a, nodes_b, interactions, counts)
    return PPResult(
        structure=structure, chain_a=chain_a, chain_b=chain_b,
        nodes_a=nodes_a, nodes_b=nodes_b, interactions=interactions,
        interface_a=interface_a, interface_b=interface_b,
        table_rows=rows, residue_rows=residue_rows, residue_stats=residue_stats,
        summary=summary, stats=stats, params=params,
    )


def _pp_summary(
    structure: Structure,
    chain_a: str,
    chain_b: str,
    interface_a: list[Residue],
    interface_b: list[Residue],
    nodes_a: list[PPNode],
    nodes_b: list[PPNode],
    interactions: list[RawInteraction],
    counts: dict[str, int],
) -> str:
    ca = chain_a.strip() or "(空白)"
    cb = chain_b.strip() or "(空白)"
    lines = [
        f"结构文件：{structure.pdb_id}",
        f"分析链：链 {ca}  vs  链 {cb}",
        f"界面残基：链 {ca} {len(interface_a)} 个，链 {cb} {len(interface_b)} 个",
        f"参与相互作用的残基：链 {ca} {len(nodes_a)} 个，链 {cb} {len(nodes_b)} 个",
        "",
        f"氢键：{counts['hbond']}    盐桥：{counts['salt']}    "
        f"π–π 堆积：{counts['pi']}    疏水接触：{counts['hydrophobic']}",
    ]
    if interactions:
        min_it = min(interactions, key=lambda x: x.distance)
        partner = min_it.res_b.label_spaced if min_it.res_b else "—"
        lines.append(f"最短接触：{min_it.distance:.2f} Å  "
                     f"（{min_it.res_a.label_spaced} — {partner}）")
    else:
        lines.append("未检测到符合阈值的界面相互作用，可适当放宽阈值。")

    pair_counts: dict[tuple[str, str], int] = {}
    for it in interactions:
        if it.res_b is None:
            continue
        key = (it.res_a.label, it.res_b.label)
        pair_counts[key] = pair_counts.get(key, 0) + 1
    top = sorted(pair_counts.items(), key=lambda kv: -kv[1])[:8]
    if top:
        lines.append("")
        lines.append("相互作用最密集的残基对：")
        for (ra, rb), n in top:
            lines.append(f"  {ra} — {rb}：{n} 项")
    return "\n".join(lines)
