"""溶剂可及表面积（SASA）与结合埋藏面积（BSA）计算。

采用 Shrake-Rupley 数值积分：
- 探针半径 1.4 Å
- 原子范德华半径加探针半径后生成测试球面
- 球面点被邻近原子遮挡的比例决定暴露面积

仅用于相对比较与描述性分析，不同半径/采样密度会带来小幅差异。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pdbio import Atom

VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "S": 1.80, "P": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "SE": 1.90, "B": 1.92, "SI": 2.10, "FE": 1.80, "ZN": 1.39,
    "MG": 1.73, "MN": 1.73, "CA": 1.94, "CU": 1.40, "NI": 1.63,
    "CO": 1.70, "CD": 1.58, "HG": 1.55, "NA": 2.27, "K": 2.75,
}

PROBE = 1.4


def atom_radius(atom: Atom) -> float:
    return VDW_RADII.get(atom.element.upper(), 1.70)


def _fibonacci_sphere(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
    return np.column_stack((
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ))


_SPHERE_CACHE: dict[int, np.ndarray] = {}


def _sphere(n: int) -> np.ndarray:
    if n not in _SPHERE_CACHE:
        _SPHERE_CACHE[n] = _fibonacci_sphere(n)
    return _SPHERE_CACHE[n]


def sasa_per_atom(atoms: list[Atom], n_points: int = 96,
                  probe: float = PROBE) -> np.ndarray:
    """返回每个原子的 SASA（Å²）。"""
    if not atoms:
        return np.zeros(0)
    coords = np.array([a.coord for a in atoms], dtype=float)
    radii = np.array([atom_radius(a) + probe for a in atoms], dtype=float)
    n = len(atoms)
    if n == 1:
        return 4.0 * np.pi * radii ** 2

    max_r = float(radii.max())
    cell = max(max_r * 2.0, 3.0)
    keys = np.floor(coords / cell).astype(int)
    bins: dict[tuple, list[int]] = {}
    for i, key in enumerate(map(tuple, keys)):
        bins.setdefault(key, []).append(i)

    unit = _sphere(n_points)
    areas = np.zeros(n, dtype=float)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1)
               for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    for i in range(n):
        ax, ay, az = keys[i]
        candidates: list[int] = []
        for dx, dy, dz in offsets:
            candidates.extend(bins.get((ax + dx, ay + dy, az + dz), ()))
        if not candidates:
            areas[i] = 4.0 * np.pi * radii[i] ** 2
            continue
        cand = np.array(candidates, dtype=int)
        cand = cand[cand != i]
        if len(cand) == 0:
            areas[i] = 4.0 * np.pi * radii[i] ** 2
            continue
        d = np.linalg.norm(coords[cand] - coords[i], axis=1)
        cand = cand[d < radii[i] + radii[cand]]
        if len(cand) == 0:
            areas[i] = 4.0 * np.pi * radii[i] ** 2
            continue
        points = coords[i] + radii[i] * unit
        exposed = np.ones(n_points, dtype=bool)
        for j in cand:
            delta = points - coords[j]
            exposed &= np.einsum("ij,ij->i", delta, delta) >= radii[j] ** 2
            if not exposed.any():
                break
        areas[i] = 4.0 * np.pi * radii[i] ** 2 * float(exposed.mean())
    return areas


@dataclass
class BindingSASA:
    sasa_a_free: float
    sasa_b_free: float
    sasa_complex: float
    interface_area: float
    buried_a: float
    buried_b: float
    per_atom_buried_a: np.ndarray
    per_atom_buried_b: np.ndarray
    per_atom_complex_a: np.ndarray
    per_atom_complex_b: np.ndarray
    probe: float = PROBE
    n_points: int = 96


def binding_sasa(atoms_a: list[Atom], atoms_b: list[Atom],
                 n_points: int = 96, probe: float = PROBE,
                 progress=None) -> BindingSASA:
    """计算两组原子的结合埋藏面积与逐原子埋藏面积。"""
    def report(msg: str):
        if progress:
            progress(msg)

    report("SASA: 游离 A")
    sasa_a = sasa_per_atom(atoms_a, n_points=n_points, probe=probe)
    report("SASA: 游离 B")
    sasa_b = sasa_per_atom(atoms_b, n_points=n_points, probe=probe)
    report("SASA: 复合物")
    combined = list(atoms_a) + list(atoms_b)
    sasa_complex_all = sasa_per_atom(combined, n_points=n_points, probe=probe)
    sasa_complex_a = sasa_complex_all[:len(atoms_a)]
    sasa_complex_b = sasa_complex_all[len(atoms_a):]

    total_a = float(sasa_a.sum())
    total_b = float(sasa_b.sum())
    total_complex = float(sasa_complex_all.sum())
    buried_a = float((sasa_a - sasa_complex_a).sum())
    buried_b = float((sasa_b - sasa_complex_b).sum())
    area = (total_a + total_b - total_complex) / 2.0
    return BindingSASA(
        sasa_a_free=total_a, sasa_b_free=total_b, sasa_complex=total_complex,
        interface_area=area, buried_a=buried_a, buried_b=buried_b,
        per_atom_buried_a=sasa_a - sasa_complex_a,
        per_atom_buried_b=sasa_b - sasa_complex_b,
        per_atom_complex_a=sasa_complex_a,
        per_atom_complex_b=sasa_complex_b,
        probe=probe, n_points=n_points,
    )
