"""相互作用检测：氢键、疏水接触、盐桥、π–π 堆积。

几何判据参考 PLIP 等工具的常用阈值：
- 氢键：重原子距离 <= 3.5 A，D-H...A 角度 >= 120 度（无显式氢时用虚拟氢）
- 疏水：C/S 原子距离 <= 4.0 A
- 盐桥：正负电荷基团原子距离 <= 4.0 A
- π 堆积：芳环质心距离 <= 5.5 A，环平面近似平行或 T 型
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import dist as _dist

import numpy as np

from . import chemdata
from .pdbio import Atom, Residue


@dataclass
class InteractionParams:
    hbond_dist: float = 3.5
    hbond_angle: float = 100.0
    hydrophobic_dist: float = 4.0
    salt_dist: float = 4.0
    pi_dist: float = 5.5
    pi_enabled: bool = True
    interface_dist: float = 4.0
    metal_dist: float = 2.9
    metal_enabled: bool = True


@dataclass
class RawInteraction:
    kind: str                       # hbond / hydrophobic / salt / pi
    res_a: Residue
    atom_a: str | None
    coord_a: np.ndarray
    atom_b: str | None
    coord_b: np.ndarray
    res_b: Residue | None
    distance: float
    angle: float | None = None
    contacts: int = 1
    note: str = ""


# ---------------------------------------------------------------------------
# 基础几何
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3)
    return v / n


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1)); n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _ring_geometry(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    radius = float(np.mean(np.linalg.norm(centered, axis=1)))
    return center, normal, radius


def _ring_planarity(coords: np.ndarray) -> float:
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    return float(np.max(np.abs(centered @ normal)))


def _has_atom(res: Residue, name: str) -> bool:
    return res.atom(name) is not None


def _protein_ring_atoms(res: Residue) -> list[tuple[str, np.ndarray]]:
    rings = []
    templates = chemdata.AROMATIC_RINGS.get(res.resname.strip().upper(), [])
    for names in templates:
        coords = []
        ok = True
        for nm in names:
            atom = res.atom(nm)
            if atom is None:
                ok = False
                break
            coords.append(atom.coord)
        if ok:
            rings.append((names, np.array(coords)))
    return rings


# ---------------------------------------------------------------------------
# 配体化学环境
# ---------------------------------------------------------------------------

@dataclass
class LigandChem:
    atoms: list[Atom]
    bonds: list[tuple[int, int]] = field(default_factory=list)
    neighbors: dict[int, list[int]] = field(default_factory=dict)
    donors: dict[int, bool] = field(default_factory=dict)
    acceptors: dict[int, bool] = field(default_factory=dict)
    charges: dict[int, int] = field(default_factory=dict)
    rings: list[list[int]] = field(default_factory=list)


def _bonded(i: int, j: int, atoms: list[Atom]) -> bool:
    a, b = atoms[i], atoms[j]
    limit = (chemdata.covalent_radius(a.element) + chemdata.covalent_radius(b.element)) * 1.18
    return float(np.linalg.norm(a.coord - b.coord)) <= max(limit, 1.9)


def _find_cycles(n: int, adj: dict[int, list[int]]) -> list[list[int]]:
    found: set[tuple[int, ...]] = set()
    for start in range(n):
        stack = [(start, [start])]
        while stack:
            cur, path = stack.pop()
            if len(path) > 6:
                continue
            for nb in adj.get(cur, []):
                if nb == start and 5 <= len(path) <= 6:
                    found.add(tuple(sorted(path)))
                elif nb > start and nb not in path:
                    stack.append((nb, path + [nb]))
    return [list(r) for r in sorted(found)]


def analyze_ligand(ligand: Residue) -> LigandChem:
    atoms = ligand.heavy_atoms
    chem = LigandChem(atoms=atoms)
    n = len(atoms)
    for i in range(n):
        chem.neighbors[i] = []
    for i, j in combinations(range(n), 2):
        if _bonded(i, j, atoms):
            chem.bonds.append((i, j))
            chem.neighbors[i].append(j)
            chem.neighbors[j].append(i)

    chem.rings = _find_cycles(n, chem.neighbors)

    # 显式氢
    explicit_h: dict[int, int] = {}
    h_atoms = [a for a in ligand.atoms if a.is_hydrogen]
    for i, atom in enumerate(atoms):
        cnt = 0
        for h in h_atoms:
            if float(np.linalg.norm(h.coord - atom.coord)) <= 1.35:
                cnt += 1
        explicit_h[i] = cnt

    aromatic_atoms: set[int] = set()
    for ring in chem.rings:
        coords = np.array([atoms[i].coord for i in ring])
        if _ring_planarity(coords) > 0.20:
            continue
        mean_len = 0.0
        cnt = 0
        for i, j in zip(ring, ring[1:] + ring[:1]):
            mean_len += _dist(atoms[i].coord, atoms[j].coord)
            cnt += 1
        mean_len /= max(cnt, 1)
        if mean_len < 1.46:
            aromatic_atoms.update(ring)

    for i, atom in enumerate(atoms):
        el = atom.element.upper()
        deg = len(chem.neighbors[i])
        donor = False
        acceptor = False
        if el in ("N", "O", "S"):
            if explicit_h.get(i):
                donor = True
            if el == "N":
                if i in aromatic_atoms:
                    donor = bool(explicit_h.get(i))
                    acceptor = not donor
                else:
                    bonded_carbonyl = False
                    for j in chem.neighbors[i]:
                        nb = atoms[j]
                        if nb.element == "C":
                            for k in chem.neighbors[j]:
                                if k == i:
                                    continue
                                if atoms[k].element == "O" and _dist(atoms[j].coord, atoms[k].coord) < 1.32:
                                    bonded_carbonyl = True
                    if deg <= 2:
                        donor = donor or True
                        acceptor = not bonded_carbonyl
                    else:
                        donor = False
                        acceptor = True
            elif el == "O":
                acceptor = True
                if deg == 1:
                    j = chem.neighbors[i][0]
                    nb = atoms[j]
                    d = _dist(atom.coord, nb.coord)
                    if nb.element == "C" and 1.34 < d < 1.55 and not explicit_h.get(i):
                        donor = True          # 羟基氧
                    elif explicit_h.get(i):
                        donor = True
                elif explicit_h.get(i):
                    donor = True
            else:  # S
                acceptor = True
                if deg == 1:
                    donor = True
                elif explicit_h.get(i):
                    donor = True
        chem.donors[i] = donor
        chem.acceptors[i] = acceptor
        chem.charges[i] = 0

    # 带电基团
    for i, atom in enumerate(atoms):
        el = atom.element.upper()
        nbrs = chem.neighbors[i]
        if el == "C":
            oxy = [j for j in nbrs if atoms[j].element == "O"
                   and 1.14 < _dist(atom.coord, atoms[j].coord) <= 1.38]
            if len(oxy) >= 2:
                for j in oxy:
                    chem.charges[j] = -1
            nitro = [j for j in nbrs if atoms[j].element == "N"]
            if len(nitro) >= 3:
                for j in nitro:
                    if _dist(atom.coord, atoms[j].coord) < 1.38:
                        chem.charges[j] = 1
        elif el == "P":
            term = [j for j in nbrs if len(chem.neighbors[j]) == 1]
            if len(term) >= 2:
                for j in term:
                    chem.charges[j] = -1
        elif el == "S":
            oxy = [j for j in nbrs if atoms[j].element == "O"]
            term = [j for j in oxy if len(chem.neighbors[j]) == 1]
            if len(oxy) >= 3 and len(term) >= 2:
                for j in term:
                    chem.charges[j] = -1
        elif el == "N":
            if len(nbrs) >= 4:
                chem.charges[i] = 1

    # 与金属配位的原子不再参与氢键（例如血红素吡咯氮、金属配位羧基氧）
    metal_idx = [i for i, a in enumerate(atoms)
                 if a.element.upper() in chemdata.METAL_ELEMENTS]
    if metal_idx:
        for i, atom in enumerate(atoms):
            if i in metal_idx:
                continue
            if any(float(np.linalg.norm(atom.coord - atoms[m].coord)) <= 2.6
                   for m in metal_idx):
                chem.donors[i] = False
                chem.acceptors[i] = False
    return chem


# ---------------------------------------------------------------------------
# 氢键几何
# ---------------------------------------------------------------------------

def _atom_neighbors(res: Residue, atom: Atom, max_d: float = 2.1) -> list[np.ndarray]:
    out = []
    for other in res.atoms:
        if other is atom or other.is_hydrogen:
            continue
        if float(np.linalg.norm(other.coord - atom.coord)) <= max_d:
            out.append(other.coord)
    return out


def _donor_h_positions(res: Residue, atom: Atom) -> list[np.ndarray]:
    """显式氢坐标；没有则返回空列表。"""
    out = []
    for other in res.atoms:
        if other.is_hydrogen and float(np.linalg.norm(other.coord - atom.coord)) <= 1.35:
            out.append(other.coord)
    return out


def _virtual_h(donor: Atom, neighbor_coords: list[np.ndarray], acceptor_coord: np.ndarray) -> np.ndarray | None:
    if not neighbor_coords:
        return None
    vec = np.zeros(3)
    for nb in neighbor_coords:
        vec += _unit(nb - donor.coord)
    direction = -vec
    if float(np.linalg.norm(direction)) < 1e-4:
        direction = _unit(donor.coord - acceptor_coord)
    if float(np.linalg.norm(direction)) < 1e-6:
        return None
    return donor.coord + 1.01 * _unit(direction)


def _hbond_geometry(
    donor_atom: Atom,
    acceptor_coord: np.ndarray,
    neighbor_coords: list[np.ndarray],
    h_positions: list[np.ndarray],
) -> tuple[float, float] | None:
    """返回 (D...A 距离, D-H...A 角度)。角度取得分最优的氢。"""
    d = float(np.linalg.norm(donor_atom.coord - acceptor_coord))
    if d < 1.2:
        return None
    h_list = list(h_positions)
    if not h_list:
        vh = _virtual_h(donor_atom, neighbor_coords, acceptor_coord)
        if vh is not None:
            h_list.append(vh)
    if not h_list:
        # 无法确定氢的方向（如孤立离子），不判定为氢键
        return None
    best_angle = -1.0
    for h in h_list:
        ang = _angle_deg(donor_atom.coord - h, acceptor_coord - h)
        best_angle = max(best_angle, ang)
    return d, best_angle


# ---------------------------------------------------------------------------
# 邻域搜索
# ---------------------------------------------------------------------------

def grid_neighbors(coords_a: np.ndarray, coords_b: np.ndarray, cutoff: float):
    """返回距离 <= cutoff 的原子索引对 (ia, ib) 及距离。"""
    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0)
    cell = max(cutoff, 1.0)
    b_cells: dict[tuple[int, int, int], list[int]] = {}
    b_keys = np.floor(coords_b / cell).astype(int)
    for j, key in enumerate(map(tuple, b_keys)):
        b_cells.setdefault(key, []).append(j)
    a_keys = np.floor(coords_a / cell).astype(int)
    ia_list: list[int] = []
    ib_list: list[int] = []
    d_list: list[float] = []
    cut2 = cutoff * cutoff
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for i, key in enumerate(map(tuple, a_keys)):
        ax, ay, az = key
        for dx, dy, dz in offsets:
            ids = b_cells.get((ax + dx, ay + dy, az + dz))
            if not ids:
                continue
            diff = coords_b[ids] - coords_a[i]
            d2 = np.einsum("ij,ij->i", diff, diff)
            mask = d2 <= cut2
            if not np.any(mask):
                continue
            sel = np.asarray(ids)[mask]
            ia_list.extend([i] * len(sel))
            ib_list.extend(sel.tolist())
            d_list.extend(np.sqrt(d2[mask]).tolist())
    return np.array(ia_list, dtype=int), np.array(ib_list, dtype=int), np.array(d_list)


# ---------------------------------------------------------------------------
# 氢键检测（配体模式）
# ---------------------------------------------------------------------------

def detect_pl_hbonds(
    residues: list[Residue],
    lig: Residue,
    chem: LigandChem,
    params: InteractionParams,
) -> list[RawInteraction]:
    out: list[RawInteraction] = []
    lig_atoms = chem.atoms
    lig_coords = np.array([a.coord for a in lig_atoms])
    metal_coords = [a.coord for a in lig_atoms
                    if a.element.upper() in chemdata.METAL_ELEMENTS]
    for res in residues:
        prot_atoms = [a for a in res.atoms if a.is_heavy]
        if not prot_atoms:
            continue
        pcoords = np.array([a.coord for a in prot_atoms])
        ia, ib, dists = grid_neighbors(pcoords, lig_coords, params.hbond_dist)
        for pi, li, d in zip(ia, ib, dists):
            patom = prot_atoms[int(pi)]
            latom = lig_atoms[int(li)]
            if metal_coords and any(
                float(np.linalg.norm(patom.coord - mc)) <= 2.7 for mc in metal_coords
            ):
                continue
            if patom.element not in ("N", "O", "S") or latom.element not in ("N", "O", "S"):
                continue
            if d < 1.9:
                continue
            p_don, p_acc = chemdata.protein_donor_acceptor(res.resname, patom.name)
            l_don = chem.donors.get(int(li), False)
            l_acc = chem.acceptors.get(int(li), False)
            best: tuple[float, float] | None = None
            if p_don and l_acc:
                geo = _hbond_geometry(
                    patom, latom.coord,
                    _atom_neighbors(res, patom),
                    _donor_h_positions(res, patom),
                )
                if geo and geo[1] >= params.hbond_angle:
                    best = geo
            if l_don and p_acc:
                nb_coords = [lig_atoms[j].coord for j in chem.neighbors.get(int(li), [])]
                h_pos = [a.coord for a in lig.atoms
                         if a.is_hydrogen and float(np.linalg.norm(a.coord - latom.coord)) <= 1.35]
                geo = _hbond_geometry(latom, patom.coord, nb_coords, h_pos)
                if geo and geo[1] >= params.hbond_angle:
                    if best is None or geo[1] > best[1]:
                        best = geo
            if best is None:
                continue
            out.append(RawInteraction(
                kind="hbond", res_a=res, atom_a=patom.name, coord_a=patom.coord,
                atom_b=latom.name, coord_b=latom.coord, res_b=None,
                distance=best[0], angle=best[1],
            ))
    return out


def detect_pl_hydrophobic(
    residues: list[Residue],
    lig: Residue,
    chem: LigandChem,
    params: InteractionParams,
) -> list[RawInteraction]:
    out: list[RawInteraction] = []
    lig_idx = [i for i, a in enumerate(chem.atoms)
               if a.element.upper() in ("C", "S")]
    if not lig_idx:
        return out
    lig_coords = np.array([chem.atoms[i].coord for i in lig_idx])
    for res in residues:
        prot = [a for a in res.atoms if a.is_heavy
                and chemdata.is_hydrophobic_atom(res.resname, a.name, a.element)]
        if not prot:
            continue
        pcoords = np.array([a.coord for a in prot])
        ia, ib, dists = grid_neighbors(pcoords, lig_coords, params.hydrophobic_dist)
        if len(ia) == 0:
            continue
        pairs = [(float(d), prot[int(i)], chem.atoms[lig_idx[int(j)]]) for i, j, d in zip(ia, ib, dists)]
        pairs = [p for p in pairs if p[0] >= 1.8]
        if not pairs:
            continue
        pairs.sort(key=lambda p: p[0])
        pa = np.mean([p[1].coord for p in pairs], axis=0)
        pb = np.mean([p[2].coord for p in pairs], axis=0)
        out.append(RawInteraction(
            kind="hydrophobic", res_a=res, atom_a=None, coord_a=pa,
            atom_b=None, coord_b=pb, res_b=None,
            distance=pairs[0][0], contacts=len(pairs),
        ))
    return out


def detect_pl_salt(
    residues: list[Residue],
    lig: Residue,
    chem: LigandChem,
    params: InteractionParams,
) -> list[RawInteraction]:
    out: list[RawInteraction] = []
    lig_pos = [i for i, c in chem.charges.items() if c > 0]
    lig_neg = [i for i, c in chem.charges.items() if c < 0]
    lig_coords = {i: chem.atoms[i].coord for i in lig_pos + lig_neg}
    for res in residues:
        for atom in res.atoms:
            if atom.is_hydrogen:
                continue
            q = chemdata.protein_charge(res.resname, atom.name)
            if q == 0:
                continue
            targets = lig_neg if q > 0 else lig_pos
            for li in targets:
                d = float(np.linalg.norm(atom.coord - lig_coords[li]))
                if d <= params.salt_dist:
                    out.append(RawInteraction(
                        kind="salt", res_a=res, atom_a=atom.name, coord_a=atom.coord,
                        atom_b=chem.atoms[li].name, coord_b=lig_coords[li], res_b=None,
                        distance=d,
                    ))
    return out


def detect_pl_metal(
    residues: list[Residue],
    lig: Residue,
    chem: LigandChem,
    params: InteractionParams,
) -> list[RawInteraction]:
    """金属配位键：配体中的金属离子与蛋白 N/O/S 原子。"""
    if not params.metal_enabled:
        return []
    metals = [(i, a) for i, a in enumerate(chem.atoms)
              if a.element.upper() in chemdata.METAL_ELEMENTS]
    if not metals:
        return []
    out: list[RawInteraction] = []
    for res in residues:
        contacts = []
        for atom in res.atoms:
            if atom.is_hydrogen or atom.element.upper() not in ("N", "O", "S"):
                continue
            for mi, matom in metals:
                d = float(np.linalg.norm(atom.coord - matom.coord))
                if d <= params.metal_dist:
                    contacts.append((d, atom, mi, matom))
        if not contacts:
            continue
        contacts.sort(key=lambda c: c[0])
        best = contacts[0]
        out.append(RawInteraction(
            kind="metal", res_a=res, atom_a=best[1].name, coord_a=best[1].coord,
            atom_b=best[3].name, coord_b=best[3].coord, res_b=None,
            distance=best[0], contacts=len(contacts), note="配位",
        ))
    return out


def detect_pl_pi(
    residues: list[Residue],
    lig: Residue,
    chem: LigandChem,
    params: InteractionParams,
) -> list[RawInteraction]:
    if not params.pi_enabled or not chem.rings:
        return []
    out: list[RawInteraction] = []
    lig_rings = []
    for ring in chem.rings:
        coords = np.array([chem.atoms[i].coord for i in ring])
        if _ring_planarity(coords) > 0.20:
            continue
        center, normal, _ = _ring_geometry(coords)
        lig_rings.append((center, normal))
    if not lig_rings:
        return []
    for res in residues:
        for names, coords in _protein_ring_atoms(res):
            pc, pn, _ = _ring_geometry(coords)
            if _ring_planarity(coords) > 0.20:
                continue
            for lc, ln in lig_rings:
                dvec = lc - pc
                d = float(np.linalg.norm(dvec))
                if d > params.pi_dist or d < 2.5:
                    continue
                theta = _angle_deg(pn, ln)
                theta = min(theta, 180.0 - theta)
                if theta <= 30.0:
                    geometry = "平行"
                elif theta >= 60.0:
                    geometry = "T 型"
                else:
                    continue
                offset = float(np.linalg.norm(dvec - np.dot(dvec, pn) * pn))
                if geometry == "平行" and offset > 2.8:
                    continue
                if geometry == "T 型":
                    vertical = abs(float(np.dot(dvec, pn)))
                    if offset > 4.5 or vertical > 2.5:
                        continue
                out.append(RawInteraction(
                    kind="pi", res_a=res, atom_a="ring", coord_a=pc,
                    atom_b="ring", coord_b=lc, res_b=None,
                    distance=d, angle=theta, note=geometry,
                ))
    return out


# ---------------------------------------------------------------------------
# 蛋白-蛋白
# ---------------------------------------------------------------------------

def protein_interface_residues(
    chain_a: list[Residue], chain_b: list[Residue], cutoff: float
) -> tuple[list[Residue], list[Residue]]:
    atoms_a: list[Atom] = []
    map_a: list[Residue] = []
    for res in chain_a:
        for atom in res.atoms:
            if atom.is_heavy:
                atoms_a.append(atom)
                map_a.append(res)
    atoms_b: list[Atom] = []
    map_b: list[Residue] = []
    for res in chain_b:
        for atom in res.atoms:
            if atom.is_heavy:
                atoms_b.append(atom)
                map_b.append(res)
    if not atoms_a or not atoms_b:
        return [], []
    ca = np.array([a.coord for a in atoms_a])
    cb = np.array([a.coord for a in atoms_b])
    ia, ib, _ = grid_neighbors(ca, cb, cutoff)
    set_a = {map_a[int(i)] for i in ia}
    set_b = {map_b[int(j)] for j in ib}
    order_a = {r.key: k for k, r in enumerate(chain_a)}
    order_b = {r.key: k for k, r in enumerate(chain_b)}
    return (
        sorted(set_a, key=lambda r: order_a.get(r.key, 0)),
        sorted(set_b, key=lambda r: order_b.get(r.key, 0)),
    )


def detect_pp_hbonds(
    residues_a: list[Residue], residues_b: list[Residue], params: InteractionParams
) -> list[RawInteraction]:
    out: list[RawInteraction] = []
    for ra in residues_a:
        atoms_a = [x for x in ra.atoms if x.is_heavy]
        if not atoms_a:
            continue
        for rb in residues_b:
            atoms_b = [x for x in rb.atoms if x.is_heavy]
            if not atoms_b:
                continue
            ca = np.array([x.coord for x in atoms_a])
            cb = np.array([x.coord for x in atoms_b])
            ia, ib, dists = grid_neighbors(ca, cb, params.hbond_dist)
            for i, j, d in zip(ia, ib, dists):
                a = atoms_a[int(i)]
                b = atoms_b[int(j)]
                if a.element not in ("N", "O", "S") or b.element not in ("N", "O", "S"):
                    continue
                if d < 1.9:
                    continue
                if a.element == "S" and b.element == "S" and d < 2.5:
                    continue  # 二硫键
                a_don, a_acc = chemdata.protein_donor_acceptor(ra.resname, a.name)
                b_don, b_acc = chemdata.protein_donor_acceptor(rb.resname, b.name)
                best = None
                if a_don and b_acc:
                    geo = _hbond_geometry(a, b.coord, _atom_neighbors(ra, a), _donor_h_positions(ra, a))
                    if geo and geo[1] >= params.hbond_angle:
                        best = geo
                if b_don and a_acc:
                    geo = _hbond_geometry(b, a.coord, _atom_neighbors(rb, b), _donor_h_positions(rb, b))
                    if geo and geo[1] >= params.hbond_angle:
                        if best is None or geo[1] > best[1]:
                            best = (geo[0], geo[1])
                            # 保持供体在 b 侧，原子顺序不交换，仅记录几何
                if best is None:
                    continue
                out.append(RawInteraction(
                    kind="hbond", res_a=ra, atom_a=a.name, coord_a=a.coord,
                    atom_b=b.name, coord_b=b.coord, res_b=rb,
                    distance=best[0], angle=best[1],
                ))
    return out


def detect_pp_hydrophobic(
    residues_a: list[Residue], residues_b: list[Residue], params: InteractionParams
) -> list[RawInteraction]:
    out: list[RawInteraction] = []
    for ra in residues_a:
        pa = [x for x in ra.atoms if x.is_heavy
              and chemdata.is_hydrophobic_atom(ra.resname, x.name, x.element)]
        if not pa:
            continue
        ca = np.array([x.coord for x in pa])
        for rb in residues_b:
            pb = [x for x in rb.atoms if x.is_heavy
                  and chemdata.is_hydrophobic_atom(rb.resname, x.name, x.element)]
            if not pb:
                continue
            cb = np.array([x.coord for x in pb])
            ia, ib, dists = grid_neighbors(ca, cb, params.hydrophobic_dist)
            if len(ia) == 0:
                continue
            pairs = [(float(d), pa[int(i)], pb[int(j)]) for i, j, d in zip(ia, ib, dists)]
            pairs = [p for p in pairs if p[0] >= 1.8]
            if not pairs:
                continue
            pairs.sort(key=lambda p: p[0])
            mean_a = np.mean([p[1].coord for p in pairs], axis=0)
            mean_b = np.mean([p[2].coord for p in pairs], axis=0)
            out.append(RawInteraction(
                kind="hydrophobic", res_a=ra, atom_a=None, coord_a=mean_a,
                atom_b=None, coord_b=mean_b, res_b=rb,
                distance=pairs[0][0], contacts=len(pairs),
            ))
    return out


def detect_pp_salt(
    residues_a: list[Residue], residues_b: list[Residue], params: InteractionParams
) -> list[RawInteraction]:
    out: list[RawInteraction] = []
    for ra in residues_a:
        for a in ra.atoms:
            if a.is_hydrogen:
                continue
            qa = chemdata.protein_charge(ra.resname, a.name)
            if qa == 0:
                continue
            for rb in residues_b:
                for b in rb.atoms:
                    if b.is_hydrogen:
                        continue
                    qb = chemdata.protein_charge(rb.resname, b.name)
                    if qb == 0 or qa * qb >= 0:
                        continue
                    d = float(np.linalg.norm(a.coord - b.coord))
                    if d <= params.salt_dist:
                        out.append(RawInteraction(
                            kind="salt", res_a=ra, atom_a=a.name, coord_a=a.coord,
                            atom_b=b.name, coord_b=b.coord, res_b=rb, distance=d,
                        ))
    return out


def detect_pp_pi(
    residues_a: list[Residue], residues_b: list[Residue], params: InteractionParams
) -> list[RawInteraction]:
    if not params.pi_enabled:
        return []
    out: list[RawInteraction] = []
    rings_a = []
    for ra in residues_a:
        for _, coords in _protein_ring_atoms(ra):
            if _ring_planarity(coords) <= 0.20:
                center, normal, _ = _ring_geometry(coords)
                rings_a.append((ra, center, normal))
    rings_b = []
    for rb in residues_b:
        for _, coords in _protein_ring_atoms(rb):
            if _ring_planarity(coords) <= 0.20:
                center, normal, _ = _ring_geometry(coords)
                rings_b.append((rb, center, normal))
    for ra, ca, na in rings_a:
        for rb, cb, nb in rings_b:
            dvec = cb - ca
            d = float(np.linalg.norm(dvec))
            if d > params.pi_dist or d < 2.5:
                continue
            theta = _angle_deg(na, nb)
            theta = min(theta, 180.0 - theta)
            if theta <= 30.0:
                geometry = "平行"
            elif theta >= 60.0:
                geometry = "T 型"
            else:
                continue
            offset = float(np.linalg.norm(dvec - np.dot(dvec, na) * na))
            if geometry == "平行" and offset > 2.8:
                continue
            if geometry == "T 型":
                vertical = abs(float(np.dot(dvec, na)))
                if offset > 4.5 or vertical > 2.5:
                    continue
            out.append(RawInteraction(
                kind="pi", res_a=ra, atom_a="ring", coord_a=ca,
                atom_b="ring", coord_b=cb, res_b=rb,
                distance=d, angle=theta, note=geometry,
            ))
    return out


def merge_hbond_salt(interactions: list[RawInteraction]) -> list[RawInteraction]:
    """同一原子对的氢键若同时构成盐桥，只保留盐桥，避免重复。"""
    salt_keys = {
        (i.res_a.key, i.atom_a, i.atom_b, i.res_b.key if i.res_b else None)
        for i in interactions if i.kind == "salt"
    }
    out = []
    for i in interactions:
        if i.kind == "hbond":
            key = (i.res_a.key, i.atom_a, i.atom_b, i.res_b.key if i.res_b else None)
            key_rev = (i.res_a.key, i.atom_b, i.atom_a, i.res_b.key if i.res_b else None)
            if key in salt_keys or key_rev in salt_keys:
                continue
        out.append(i)
    return out
