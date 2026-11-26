"""PDB 结构解析模块。

只依赖标准库与 numpy，支持 .pdb / .ent / .pdb.gz。
仅读取第一个 MODEL，自动去除重复 altLoc。
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass, field

import numpy as np

WATER_NAMES = {"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"}

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL", "HSD", "HSE", "HSP", "CYX", "ASH", "GLH", "LYN",
}

_AA_DISPLAY = {
    "ALA": "Ala", "ARG": "Arg", "ASN": "Asn", "ASP": "Asp", "CYS": "Cys",
    "GLN": "Gln", "GLU": "Glu", "GLY": "Gly", "HIS": "His", "ILE": "Ile",
    "LEU": "Leu", "LYS": "Lys", "MET": "Met", "PHE": "Phe", "PRO": "Pro",
    "SER": "Ser", "THR": "Thr", "TRP": "Trp", "TYR": "Tyr", "VAL": "Val",
    "MSE": "Mse", "SEC": "Sec", "PYL": "Pyl", "HSD": "His", "HSE": "His",
    "HSP": "His", "CYX": "Cys", "ASH": "Asp", "GLH": "Glu", "LYN": "Lys",
}

# 常见双字母元素（用于元素列缺失时的推断）
_TWO_LETTER = {
    "FE", "ZN", "NA", "CL", "BR", "MG", "MN", "CA", "CU", "NI", "CO", "CD",
    "HG", "SE", "SI", "SR", "MO", "W", "AU", "PT", "AG", "LI", "AL", "GA",
    "GE", "AS", "PB", "CS", "BA", "RB", "YB", "TB", "EU", "GD", "SM", "ND",
}


@dataclass
class Atom:
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    coord: np.ndarray
    element: str
    occupancy: float
    bfactor: float
    is_hetatm: bool
    index: int = 0

    @property
    def is_hydrogen(self) -> bool:
        return self.element in ("H", "D")

    @property
    def is_heavy(self) -> bool:
        return not self.is_hydrogen


@dataclass(eq=False)
class Residue:
    resname: str
    chain: str
    resseq: int
    icode: str
    is_hetero: bool
    atoms: list[Atom] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.chain, self.resseq, self.icode, self.resname)

    @property
    def is_water(self) -> bool:
        return self.resname.strip().upper() in WATER_NAMES

    @property
    def is_amino(self) -> bool:
        return self.resname.strip().upper() in STANDARD_AA

    @property
    def display_name(self) -> str:
        return _AA_DISPLAY.get(self.resname.strip().upper(), self.resname.strip())

    @property
    def label(self) -> str:
        """图注用紧凑标签：Ser195(A)"""
        chain = self.chain.strip() or "-"
        icode = self.icode.strip()
        return f"{self.display_name}{self.resseq}{icode}({chain})"

    @property
    def label_spaced(self) -> str:
        """表格用标签：Ser195 (A)"""
        chain = self.chain.strip() or "-"
        icode = self.icode.strip()
        return f"{self.display_name}{self.resseq}{icode} ({chain})"

    @property
    def heavy_atoms(self) -> list[Atom]:
        return [a for a in self.atoms if a.is_heavy]

    def atom(self, name: str) -> Atom | None:
        name = name.strip()
        for a in self.atoms:
            if a.name == name:
                return a
        return None

    def coords(self, heavy_only: bool = True) -> np.ndarray:
        atoms = self.heavy_atoms if heavy_only else self.atoms
        if not atoms:
            return np.zeros((0, 3))
        return np.array([a.coord for a in atoms])


@dataclass
class Structure:
    path: str
    title: str
    atoms: list[Atom]
    residues: list[Residue]

    @property
    def pdb_id(self) -> str:
        stem = os.path.basename(self.path)
        for ext in (".gz", ".pdb", ".ent", ".cif"):
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
        return stem.upper()

    def chains(self) -> list[str]:
        seen: list[str] = []
        for r in self.residues:
            if r.is_water:
                continue
            if r.chain not in seen:
                seen.append(r.chain)
        return seen

    def chain_residues(self, chain: str, protein_only: bool = True) -> list[Residue]:
        out = []
        for r in self.residues:
            if r.chain != chain or r.is_water:
                continue
            if protein_only and not (r.is_amino or not r.is_hetero):
                continue
            out.append(r)
        return out

    def protein_residues(self) -> list[Residue]:
        return [
            r for r in self.residues
            if not r.is_water and (r.is_amino or not r.is_hetero)
        ]

    def ligands(self) -> list[Residue]:
        """非水 HETATM 残基，按原子数从多到少排序。"""
        ligs = [r for r in self.residues if r.is_hetero and not r.is_water]
        ligs.sort(key=lambda r: len(r.heavy_atoms), reverse=True)
        return ligs

    def chain_label(self, chain: str) -> str:
        return chain.strip() or "(空白)"

    def summary(self) -> str:
        chains = [self.chain_label(c) for c in self.chains()]
        n_res = len([r for r in self.residues if not r.is_water])
        return (
            f"链: {', '.join(chains)}\n"
            f"残基: {n_res}    原子: {len(self.atoms)}"
        )


def _infer_element(name: str, resname: str, is_hetatm: bool) -> str:
    n = name.strip()
    if not n:
        return "C"
    upper = n.upper()
    if is_hetatm:
        two = upper[:2]
        if two in _TWO_LETTER:
            # 避免把蛋白质常见原子名误判为金属（如 CA 已由 ATOM 记录处理）
            if two not in ("CA", "CD", "CO", "NI", "CU", "MG", "FE", "NA", "CL"):
                return two
            if two == "CL" and not any(c.isdigit() for c in n):
                return "CL"
            if two in ("FE", "ZN", "NA", "MG", "MN", "CU", "NI", "CO", "CD", "HG") and len(n) <= 2:
                return two
    for ch in n:
        if ch.isalpha():
            return ch.upper()
    return "C"


def _altloc_score(altloc: str, occupancy: float) -> tuple[int, float]:
    if altloc == " ":
        pri = 2
    elif altloc.strip().upper() == "A":
        pri = 1
    else:
        pri = 0
    return (pri, occupancy)


def parse_pdb(path: str) -> Structure:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    opener = gzip.open if path.lower().endswith(".gz") else open
    title_parts: list[str] = []
    raw_atoms: list[dict] = []
    atom_index = 0
    saw_model = False
    model_done = False

    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            rec = line[:6].strip().upper()
            if rec == "MODEL":
                if saw_model:
                    model_done = True
                    break
                saw_model = True
                continue
            if rec == "ENDMDL":
                if saw_model:
                    break
                continue
            if rec == "TITLE":
                title_parts.append(line[10:].strip())
                continue
            if rec not in ("ATOM", "HETATM"):
                continue
            if model_done:
                break
            try:
                serial = int(line[6:11])
                name = line[12:16].strip()
                altloc = line[16]
                resname = line[17:20].strip()
                chain = line[21]
                resseq = int(line[22:26])
                icode = line[26]
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            try:
                occ = float(line[54:60])
            except ValueError:
                occ = 1.0
            try:
                bfac = float(line[60:66])
            except ValueError:
                bfac = 0.0
            element = line[76:78].strip().upper()
            if not element:
                element = _infer_element(name, resname, rec == "HETATM")
            atom_index += 1
            raw_atoms.append(dict(
                serial=serial, name=name, altloc=altloc, resname=resname,
                chain=chain, resseq=resseq, icode=icode,
                x=x, y=y, z=z, element=element, occ=occ, bfac=bfac,
                is_hetatm=(rec == "HETATM"), index=atom_index,
            ))

    if not raw_atoms:
        raise ValueError("该文件中未找到 ATOM/HETATM 记录，可能不是有效的结构文件。")
    return _assemble_structure(path, " ".join(t for t in title_parts if t).strip(), raw_atoms)


def _assemble_structure(path: str, title: str, raw_atoms: list[dict]) -> Structure:
    if not raw_atoms:
        raise ValueError("该文件中未找到可用的原子记录。")

    # 过滤 altLoc：同一原子名只保留最优构象
    best: dict[tuple, dict] = {}
    for rec in raw_atoms:
        key = (rec["chain"], rec["resseq"], rec["icode"], rec["resname"], rec["name"])
        old = best.get(key)
        if old is None:
            best[key] = rec
        else:
            if _altloc_score(rec["altloc"], rec["occ"]) > _altloc_score(old["altloc"], old["occ"]):
                best[key] = rec

    kept = sorted(best.values(), key=lambda r: r["index"])
    atoms: list[Atom] = []
    residues: list[Residue] = []
    res_map: dict[tuple, Residue] = {}
    for rec in kept:
        atom = Atom(
            serial=rec["serial"], name=rec["name"], altloc=rec["altloc"],
            resname=rec["resname"], chain=rec["chain"], resseq=rec["resseq"],
            icode=rec["icode"], coord=np.array([rec["x"], rec["y"], rec["z"]], dtype=float),
            element=rec["element"], occupancy=rec["occ"], bfactor=rec["bfac"],
            is_hetatm=rec["is_hetatm"], index=rec["index"],
        )
        atoms.append(atom)
        rkey = (atom.chain, atom.resseq, atom.icode, atom.resname)
        res = res_map.get(rkey)
        if res is None:
            res = Residue(
                resname=atom.resname, chain=atom.chain, resseq=atom.resseq,
                icode=atom.icode, is_hetero=atom.is_hetatm,
            )
            res_map[rkey] = res
            residues.append(res)
        res.atoms.append(atom)

    if not title:
        title = os.path.basename(path)
    return Structure(path=path, title=title, atoms=atoms, residues=residues)


# ---------------------------------------------------------------------------
# mmCIF / CIF
# ---------------------------------------------------------------------------

def _cif_tokens(path: str) -> list[str]:
    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    tokens: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith(";"):
            buf = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                buf.append(lines[i])
                i += 1
            i += 1
            tokens.append("\n".join(buf).strip())
            continue
        pos = 0
        length = len(line)
        while pos < length:
            ch = line[pos]
            if ch in " \t":
                pos += 1
                continue
            if ch == "#":
                break
            if ch in ("'", '"'):
                quote = ch
                pos += 1
                start = pos
                while pos < length and line[pos] != quote:
                    pos += 1
                tokens.append(line[start:pos])
                pos += 1
                continue
            start = pos
            while pos < length and line[pos] not in " \t":
                pos += 1
            tokens.append(line[start:pos])
        i += 1
    return tokens


def _cif_clean(value: str) -> str:
    v = (value or "").strip()
    return "" if v in ("?", ".") else v


def _cif_title(tokens: list[str]) -> str:
    for idx, token in enumerate(tokens):
        if token == "_struct.title" and idx + 1 < len(tokens):
            value = _cif_clean(tokens[idx + 1])
            if value:
                return " ".join(value.split())
    return ""


def _cif_atom_site(tokens: list[str]) -> tuple[list[str], list[str]]:
    total = len(tokens)
    for idx, token in enumerate(tokens):
        if token.lower() != "loop_":
            continue
        j = idx + 1
        tags: list[str] = []
        while j < total and tokens[j].startswith("_"):
            tags.append(tokens[j])
            j += 1
        if not any(tag.startswith("_atom_site.") for tag in tags):
            continue
        values: list[str] = []
        while j < total:
            token = tokens[j]
            low = token.lower()
            if (token.startswith("_") or low in ("loop_", "stop_")
                    or low.startswith("data_") or low.startswith("save_")):
                break
            values.append(token)
            j += 1
        return tags, values
    return [], []


def parse_cif(path: str) -> Structure:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    tokens = _cif_tokens(path)
    tags, values = _cif_atom_site(tokens)
    if not tags:
        raise ValueError("该 CIF 文件中未找到 _atom_site 记录。")
    fields = [tag.split(".", 1)[1] for tag in tags]
    col = {name: idx for idx, name in enumerate(fields)}
    ncol = len(fields)
    if ncol == 0:
        raise ValueError("CIF 的原子记录字段为空。")
    rows = [values[k:k + ncol] for k in range(0, len(values) - ncol + 1, ncol)]
    if not rows:
        raise ValueError("CIF 中的原子记录为空。")

    def get(row: list[str], name: str) -> str:
        idx = col.get(name)
        if idx is None or idx >= len(row):
            return ""
        return _cif_clean(row[idx])

    first_model = None
    raw_atoms: list[dict] = []
    atom_index = 0
    for row in rows:
        group = get(row, "group_PDB").upper() or "ATOM"
        if group not in ("ATOM", "HETATM"):
            continue
        model = get(row, "pdbx_PDB_model_num")
        if model:
            if first_model is None:
                first_model = model
            elif model != first_model:
                continue
        name = get(row, "auth_atom_id") or get(row, "label_atom_id")
        resname = get(row, "auth_comp_id") or get(row, "label_comp_id")
        chain = get(row, "auth_asym_id") or get(row, "label_asym_id")
        seq_text = get(row, "auth_seq_id") or get(row, "label_seq_id")
        icode = get(row, "pdbx_PDB_ins_code") or " "
        altloc = get(row, "label_alt_id") or " "
        if len(altloc) > 1:
            altloc = altloc[0]
        try:
            x = float(get(row, "Cartn_x"))
            y = float(get(row, "Cartn_y"))
            z = float(get(row, "Cartn_z"))
        except ValueError:
            continue
        if not name or not resname:
            continue
        resseq = 0
        if seq_text:
            try:
                resseq = int(seq_text)
            except ValueError:
                try:
                    resseq = int(float(seq_text))
                except ValueError:
                    digits = "".join(ch for ch in seq_text if ch.isdigit())
                    resseq = int(digits) if digits else 0
            suffix = "".join(ch for ch in seq_text if ch.isalpha())
            if suffix and icode.strip() == "":
                icode = suffix[:1]
        try:
            occ = float(get(row, "occupancy") or 1.0)
        except ValueError:
            occ = 1.0
        try:
            bfac = float(get(row, "B_iso_or_equiv") or 0.0)
        except ValueError:
            bfac = 0.0
        element = get(row, "type_symbol").upper()
        if not element:
            element = _infer_element(name, resname, group == "HETATM")
        atom_index += 1
        raw_atoms.append(dict(
            serial=atom_index, name=name, altloc=altloc, resname=resname,
            chain=chain, resseq=resseq, icode=icode,
            x=x, y=y, z=z, element=element, occ=occ, bfac=bfac,
            is_hetatm=(group == "HETATM"), index=atom_index,
        ))

    return _assemble_structure(path, _cif_title(tokens), raw_atoms)


def parse_structure(path: str) -> Structure:
    """按扩展名自动选择 PDB 或 mmCIF 解析器。"""
    low = path.lower()
    if low.endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz")):
        return parse_cif(path)
    return parse_pdb(path)


def is_structure_file(path: str) -> bool:
    low = path.lower()
    return low.endswith((
        ".pdb", ".ent", ".pdb.gz", ".ent.gz",
        ".cif", ".mmcif", ".cif.gz", ".mmcif.gz",
    ))

