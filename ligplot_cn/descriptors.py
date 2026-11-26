"""结合界面与配体的定量描述符。

包含：
- 配体：观测原子组成、质量、环数、芳香环、可旋转键、供体/受体、埋藏比例
- 蛋白–小分子：BSA、配体埋藏比例、口袋残基组成、互作密度、热点残基
- 蛋白–蛋白：界面埋藏面积（单侧/总）、界面残基组成、互作密度、热点残基
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .analysis import PLResult, PPResult
from .chemdata import METAL_ELEMENTS
from .interactions import analyze_ligand
from .pdbio import Residue
from .sasa import BindingSASA

ATOMIC_MASS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "P": 30.974, "S": 32.06, "CL": 35.45, "BR": 79.904, "I": 126.90,
    "SE": 78.97, "B": 10.81, "SI": 28.085, "FE": 55.845, "ZN": 65.38,
    "MG": 24.305, "MN": 54.938, "CA": 40.078, "CU": 63.546, "NI": 58.693,
    "CO": 58.933, "CD": 112.414, "HG": 200.592, "NA": 22.990, "K": 39.098,
}

AA_CLASS = {
    "ALA": "Hydrophobic", "VAL": "Hydrophobic", "LEU": "Hydrophobic",
    "ILE": "Hydrophobic", "MET": "Hydrophobic", "PRO": "Hydrophobic",
    "MSE": "Hydrophobic",
    "PHE": "Aromatic", "TYR": "Aromatic", "TRP": "Aromatic",
    "SER": "Polar", "THR": "Polar", "ASN": "Polar", "GLN": "Polar",
    "CYS": "Polar", "SEC": "Polar",
    "LYS": "Positive", "ARG": "Positive", "HIS": "Positive",
    "HSD": "Positive", "HSE": "Positive", "HSP": "Positive",
    "PYL": "Positive",
    "ASP": "Negative", "GLU": "Negative",
    "GLY": "Glycine",
}
CLASS_ORDER = ["Hydrophobic", "Aromatic", "Polar", "Positive", "Negative", "Glycine"]
CLASS_CN = {
    "Hydrophobic": "疏水", "Aromatic": "芳香", "Polar": "极性",
    "Positive": "正电", "Negative": "负电", "Glycine": "甘氨酸",
}
CLASS_EN = {
    "Hydrophobic": "Hydrophobic", "Aromatic": "Aromatic", "Polar": "Polar",
    "Positive": "Positive", "Negative": "Negative", "Glycine": "Glycine",
}

DOUBLE_THRESHOLD = {
    ("C", "C"): 1.42, ("C", "N"): 1.36, ("C", "O"): 1.33,
    ("C", "S"): 1.72, ("N", "N"): 1.30, ("N", "O"): 1.32,
    ("O", "P"): 1.60, ("O", "S"): 1.52,
}


@dataclass
class ClassComposition:
    counts: dict[str, int]
    percentages: dict[str, float]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


@dataclass
class LigandDescriptors:
    formula: str
    mass: float
    heavy_atoms: int
    heteroatoms: int
    rings: int
    aromatic_rings: int
    rotatable_bonds: int
    donors: int
    acceptors: int
    formal_charge: int
    halogens: int
    hydrophobic_atoms: int
    sasa_free: float
    sasa_bound: float
    buried_area: float
    burial_fraction: float
    interactions_per_atom: float
    element_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class PLAssessment:
    descriptors: LigandDescriptors
    sasa: BindingSASA
    bsa: float
    total_buried: float
    pocket_composition: ClassComposition
    background_composition: ClassComposition
    pocket_residue_bsa: dict[tuple, float]
    interaction_density: float
    hbond_density: float
    hydrophobic_fraction: float
    hotspots: list[dict]
    contact_atom_fraction: float


@dataclass
class PPAssessment:
    sasa: BindingSASA
    bsa: float
    total_buried: float
    interface_area: float
    interface_size_class: str
    composition_a: ClassComposition
    composition_b: ClassComposition
    composition_all: ClassComposition
    background_composition: ClassComposition
    residue_bsa_a: dict[tuple, float]
    residue_bsa_b: dict[tuple, float]
    interaction_density: float
    hbond_density: float
    salt_density: float
    hydrophobic_fraction: float
    hotspots: list[dict]


def residue_class(resname: str) -> str:
    return AA_CLASS.get(resname.strip().upper(), "Polar")


def composition(residues: Iterable[Residue]) -> ClassComposition:
    counts = {k: 0 for k in CLASS_ORDER}
    for residue in residues:
        counts[residue_class(residue.resname)] += 1
    total = sum(counts.values())
    percentages = {
        k: (100.0 * v / total) if total else 0.0 for k, v in counts.items()
    }
    return ClassComposition(counts=counts, percentages=percentages)


def _observed_formula(atoms) -> tuple[str, dict[str, int], float]:
    counts: Counter[str] = Counter()
    mass = 0.0
    for atom in atoms:
        el = atom.element.upper()
        counts[el] += 1
        mass += ATOMIC_MASS.get(el, 0.0)
    if not counts:
        return "", {}, 0.0
    order = []
    if "C" in counts:
        order.append("C")
        if "H" in counts:
            order.append("H")
    for el in sorted(counts):
        if el not in order:
            order.append(el)
    formula = "".join(
        el + (str(counts[el]) if counts[el] > 1 else "") for el in order
    )
    return formula, dict(counts), mass


def _ring_bonds(chem) -> set[tuple[int, int]]:
    bonds = set()
    for ring in chem.rings:
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            bonds.add((min(a, b), max(a, b)))
    return bonds


def _aromatic_ring_count(chem) -> int:
    count = 0
    atoms = chem.atoms
    for ring in chem.rings:
        coords = np.array([atoms[i].coord for i in ring])
        center = coords.mean(axis=0)
        _, _, vt = np.linalg.svd(coords - center, full_matrices=False)
        planarity = float(np.max(np.abs((coords - center) @ vt[-1])))
        mean_len = float(np.mean([
            np.linalg.norm(atoms[ring[k]].coord - atoms[ring[(k + 1) % len(ring)]].coord)
            for k in range(len(ring))
        ]))
        if planarity <= 0.20 and mean_len < 1.46:
            count += 1
    return count


def _ligand_formal_charge(chem) -> int:
    """按官能团估计形式电荷，避免把胍基三个 N 重复计为 +3。"""
    atoms = chem.atoms
    charge = 0
    for i, atom in enumerate(atoms):
        el = atom.element.upper()
        nbrs = chem.neighbors.get(i, [])
        if el == "C":
            oxy = [j for j in nbrs
                   if atoms[j].element.upper() == "O"
                   and 1.14 < float(np.linalg.norm(atom.coord - atoms[j].coord)) <= 1.38]
            if len(oxy) >= 2:
                charge -= 1
            nitro = [j for j in nbrs
                     if atoms[j].element.upper() == "N"
                     and float(np.linalg.norm(atom.coord - atoms[j].coord)) < 1.38]
            if len(nitro) >= 2:
                charge += 1
        elif el == "P":
            terminal = [j for j in nbrs
                        if atoms[j].element.upper() == "O"
                        and len(chem.neighbors.get(j, [])) == 1]
            if len(terminal) >= 2:
                charge -= 1
        elif el == "S":
            oxy = [j for j in nbrs if atoms[j].element.upper() == "O"]
            terminal = [j for j in oxy if len(chem.neighbors.get(j, [])) == 1]
            if len(oxy) >= 3 and len(terminal) >= 2:
                charge -= 1
        elif el == "N":
            if len(nbrs) >= 4:
                charge += 1
    return charge


def _rotatable_bonds(chem) -> int:
    atoms = chem.atoms
    ring_bonds = _ring_bonds(chem)
    carboxylate_c = set()
    for i, atom in enumerate(atoms):
        if atom.element.upper() != "C":
            continue
        oxy = [j for j in chem.neighbors.get(i, [])
               if atoms[j].element.upper() == "O"
               and float(np.linalg.norm(atom.coord - atoms[j].coord)) < 1.36]
        if len(oxy) >= 2:
            carboxylate_c.add(i)
    count = 0
    for i, j in chem.bonds:
        key = (min(i, j), max(i, j))
        if key in ring_bonds:
            continue
        ai, aj = atoms[i], atoms[j]
        if len(chem.neighbors.get(i, [])) < 2 or len(chem.neighbors.get(j, [])) < 2:
            continue
        if i in carboxylate_c or j in carboxylate_c:
            continue
        pair = tuple(sorted((ai.element.upper(), aj.element.upper())))
        threshold = DOUBLE_THRESHOLD.get(pair, 1.20)
        length = float(np.linalg.norm(ai.coord - aj.coord))
        if length <= threshold:
            continue
        if pair == ("C", "N"):
            carbon = i if ai.element.upper() == "C" else j
            nitrogen = j if carbon == i else i
            has_carbonyl = any(
                atoms[k].element.upper() == "O"
                and float(np.linalg.norm(atoms[carbon].coord - atoms[k].coord)) < 1.34
                for k in chem.neighbors.get(carbon, [])
            )
            if has_carbonyl and len(chem.neighbors.get(nitrogen, [])) <= 2:
                continue
        if pair == ("N", "S"):
            continue
        count += 1
    return count


def ligand_descriptors(result: PLResult, sasa: BindingSASA) -> LigandDescriptors:
    ligand = result.ligand
    chem = result.ligand_chem
    atoms = ligand.heavy_atoms
    formula, counts, mass = _observed_formula(ligand.atoms)
    hetero = sum(1 for a in atoms if a.element.upper() not in ("C", "H"))
    halogens = sum(1 for a in atoms if a.element.upper() in ("F", "CL", "BR", "I"))
    donors = sum(1 for v in chem.donors.values() if v)
    acceptors = sum(1 for v in chem.acceptors.values() if v)
    charge = _ligand_formal_charge(chem)
    hydrophobic = sum(1 for a in atoms if a.element.upper() in ("C", "S"))
    buried = sasa.buried_b
    free = max(sasa.sasa_b_free, 1e-6)
    interactions = len(result.interactions)
    return LigandDescriptors(
        formula=formula, mass=mass, heavy_atoms=len(atoms), heteroatoms=hetero,
        rings=len(chem.rings), aromatic_rings=_aromatic_ring_count(chem),
        rotatable_bonds=_rotatable_bonds(chem), donors=donors, acceptors=acceptors,
        formal_charge=charge, halogens=halogens, hydrophobic_atoms=hydrophobic,
        sasa_free=sasa.sasa_b_free, sasa_bound=float(sasa.per_atom_complex_b.sum()),
        buried_area=buried, burial_fraction=buried / free,
        interactions_per_atom=interactions / max(len(atoms), 1),
        element_counts=counts,
    )


def _protein_atom_owners(structure, residues: list[Residue]):
    atoms = []
    owners = []
    for residue in residues:
        for atom in residue.heavy_atoms:
            atoms.append(atom)
            owners.append(residue)
    return atoms, owners


def _residue_burial(buried_per_atom: np.ndarray,
                    owners: list[Residue]) -> dict[tuple, float]:
    values: dict[tuple, float] = defaultdict(float)
    for value, residue in zip(buried_per_atom, owners):
        if value > 0:
            values[residue.key] += float(value)
    return dict(values)


def _hotspots(result, bsa_map: dict[tuple, float], limit: int = 5) -> list[dict]:
    out = []
    for stat in result.residue_stats[:limit]:
        label = stat["label"]
        key = None
        for residue in (result.structure.residues):
            if residue.label_spaced == label:
                key = residue.key
                break
        out.append({
            "label": label,
            "total": stat["total"],
            "min_dist": stat["min_dist"],
            "bsa": bsa_map.get(key, 0.0) if key else 0.0,
        })
    return out


def analyze_pl_binding(result: PLResult, progress=None) -> PLAssessment:
    from .sasa import binding_sasa

    protein_residues = result.structure.protein_residues()
    protein_atoms, owners = _protein_atom_owners(result.structure, protein_residues)
    ligand_atoms = result.ligand.heavy_atoms
    sasa = binding_sasa(protein_atoms, ligand_atoms, n_points=96, progress=progress)
    descriptor = ligand_descriptors(result, sasa)
    pocket_residues = [n.residue for n in result.nodes]
    pocket_comp = composition(pocket_residues)
    background = composition(protein_residues)
    pocket_bsa = _residue_burial(sasa.per_atom_buried_a, owners)
    bsa = sasa.interface_area
    total_buried = sasa.buried_a + sasa.buried_b
    records = len(result.interactions)
    counts = result.stats
    density = 100.0 * records / max(bsa, 1.0)
    hb_density = 100.0 * counts.get("hbond", 0) / max(bsa, 1.0)
    hydrophobic_fraction = counts.get("hydrophobic", 0) / max(records, 1)
    contact_atoms = set()
    for it in result.interactions:
        if it.atom_b and it.atom_b != "ring":
            contact_atoms.add(it.atom_b)
    contact_fraction = len(contact_atoms) / max(len(ligand_atoms), 1)
    return PLAssessment(
        descriptors=descriptor, sasa=sasa, bsa=bsa, total_buried=total_buried,
        pocket_composition=pocket_comp, background_composition=background,
        pocket_residue_bsa=pocket_bsa, interaction_density=density,
        hbond_density=hb_density, hydrophobic_fraction=hydrophobic_fraction,
        hotspots=_hotspots(result, pocket_bsa),
        contact_atom_fraction=contact_fraction,
    )


def analyze_pp_binding(result: PPResult, progress=None) -> PPAssessment:
    from .sasa import binding_sasa

    residues_a = result.structure.chain_residues(result.chain_a, True)
    residues_b = result.structure.chain_residues(result.chain_b, True)
    atoms_a, owners_a = _protein_atom_owners(result.structure, residues_a)
    atoms_b, owners_b = _protein_atom_owners(result.structure, residues_b)
    sasa = binding_sasa(atoms_a, atoms_b, n_points=96, progress=progress)
    bsa = sasa.interface_area
    total_buried = sasa.buried_a + sasa.buried_b
    if total_buried < 600:
        size_class = "small"
    elif total_buried < 1200:
        size_class = "medium"
    else:
        size_class = "large"
    comp_a = composition(result.interface_a)
    comp_b = composition(result.interface_b)
    comp_all = composition(list(result.interface_a) + list(result.interface_b))
    background = composition(residues_a + residues_b)
    bsa_a = _residue_burial(sasa.per_atom_buried_a, owners_a)
    bsa_b = _residue_burial(sasa.per_atom_buried_b, owners_b)
    combined_bsa = {**bsa_a, **bsa_b}
    records = len(result.interactions)
    counts = result.stats
    density = 100.0 * records / max(bsa, 1.0)
    hb_density = 100.0 * counts.get("hbond", 0) / max(bsa, 1.0)
    salt_density = 100.0 * counts.get("salt", 0) / max(bsa, 1.0)
    hydrophobic_fraction = counts.get("hydrophobic", 0) / max(records, 1)
    return PPAssessment(
        sasa=sasa, bsa=bsa, total_buried=total_buried, interface_area=bsa,
        interface_size_class=size_class,
        composition_a=comp_a, composition_b=comp_b, composition_all=comp_all,
        background_composition=background,
        residue_bsa_a=bsa_a, residue_bsa_b=bsa_b,
        interaction_density=density, hbond_density=hb_density,
        salt_density=salt_density, hydrophobic_fraction=hydrophobic_fraction,
        hotspots=_hotspots(result, combined_bsa),
    )
