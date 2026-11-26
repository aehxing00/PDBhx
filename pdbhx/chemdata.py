"""化学模板数据：元素半径/颜色、氢键供受体、带电基团、芳香环。"""

from __future__ import annotations

COVALENT_RADII = {
    "H": 0.31, "D": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "CL": 1.02, "BR": 1.20, "I": 1.39, "SE": 1.20,
    "B": 0.84, "SI": 1.11, "FE": 1.32, "ZN": 1.22, "MG": 1.41, "MN": 1.39,
    "CA": 1.76, "NA": 1.66, "K": 2.03, "CU": 1.32, "NI": 1.24, "CO": 1.26,
    "CD": 1.44, "HG": 1.32, "LI": 1.28, "AL": 1.21, "GA": 1.22, "GE": 1.20,
    "AS": 1.19, "SR": 1.95, "MO": 1.54, "W": 1.62, "AU": 1.36, "PT": 1.36,
    "AG": 1.45, "CS": 2.44, "BA": 2.15, "RB": 2.20,
}

ELEMENT_COLORS = {
    "C": "#2F3437", "N": "#2C5AA0", "O": "#D93B30", "S": "#C99A00",
    "P": "#E07B24", "F": "#2FA84F", "CL": "#2FA84F", "BR": "#8C4A2F",
    "I": "#6B3FA0", "SE": "#C97B1E", "FE": "#B05A2A", "ZN": "#7C8B99",
    "MG": "#3C8C40", "CA": "#8A8F94", "NA": "#7B68EE", "K": "#8B5CF6",
}

ELEMENT_RADIUS = {
    "C": 0.058, "N": 0.062, "O": 0.062, "S": 0.075, "P": 0.075,
    "F": 0.055, "CL": 0.068, "BR": 0.075, "I": 0.085, "SE": 0.078,
}

AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}

# 常见配位金属（用于识别金属配位键）
METAL_ELEMENTS = {"FE", "ZN", "MG", "MN", "CA", "CU", "NI", "CO", "CD", "HG"}

# 主链供体/受体
BACKBONE_DONORS = {"N"}
BACKBONE_ACCEPTORS = {"O", "OXT", "OT1", "OT2"}

# 侧链供体（给出氢）
SIDECHAIN_DONORS = {
    "ARG": {"NE", "NH1", "NH2"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"},
    "HSD": {"ND1"}, "HSE": {"NE2"}, "HSP": {"ND1", "NE2"},
    "LYS": {"NZ"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TRP": {"NE1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
    "CYX": set(),
    "MSE": set(),
}

# 侧链受体
SIDECHAIN_ACCEPTORS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "HIS": {"ND1", "NE2"},
    "HSD": {"ND1"}, "HSE": {"NE2"}, "HSP": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "MET": {"SD"},
    "CYS": {"SG"},
    "CYX": set(),
}

POSITIVE_ATOMS = {
    "LYS": {"NZ"},
    "ARG": {"NE", "NH1", "NH2"},
    "HIS": {"ND1", "NE2"},
    "HSD": {"ND1"}, "HSE": {"NE2"}, "HSP": {"ND1", "NE2"},
}

NEGATIVE_ATOMS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}

AROMATIC_RINGS = {
    "PHE": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "TYR": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "HIS": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HSD": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HSE": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HSP": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "TRP": [
        ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
        ("CG", "CD1", "NE1", "CE2", "CD2"),
    ],
}

# 疏水原子：侧链碳/硫（排除主链）
BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT", "OT1", "OT2", "H", "HA"}


def covalent_radius(element: str) -> float:
    return COVALENT_RADII.get(element.upper(), 0.77)


def element_color(element: str) -> str:
    return ELEMENT_COLORS.get(element.upper(), "#7A7A7A")


def element_radius(element: str) -> float:
    return ELEMENT_RADIUS.get(element.upper(), 0.06)


def is_hydrophobic_atom(resname: str, atom_name: str, element: str) -> bool:
    if element.upper() not in ("C", "S"):
        return False
    if atom_name.strip() in BACKBONE_NAMES:
        return False
    return True


def protein_donor_acceptor(resname: str, atom_name: str) -> tuple[bool, bool]:
    """返回蛋白质原子是否可作供体/受体。"""
    rn = resname.strip().upper()
    an = atom_name.strip()
    donor = False
    acceptor = False
    if an == "N" and rn != "PRO":
        donor = True
    if an in BACKBONE_ACCEPTORS:
        acceptor = True
    if an in SIDECHAIN_DONORS.get(rn, set()):
        donor = True
    if an in SIDECHAIN_ACCEPTORS.get(rn, set()):
        acceptor = True
    return donor, acceptor


def protein_charge(resname: str, atom_name: str) -> int:
    rn = resname.strip().upper()
    an = atom_name.strip()
    if an in POSITIVE_ATOMS.get(rn, set()):
        return 1
    if an in NEGATIVE_ATOMS.get(rn, set()):
        return -1
    if an in ("OXT", "OT2"):
        return -1
    return 0
