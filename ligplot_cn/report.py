"""中英双语 PDF 报告生成（ReportLab，字体嵌入）。

每张图配有中文与英文图注，并附简短解读；包含 SASA/BSA、残基组成、
配体描述符、界面评价、关键表格与方法说明。
"""

from __future__ import annotations

import datetime
import os
import tempfile
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image as RLImage, KeepTogether,
    LongTable, NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer,
    Table, TableStyle,
)
from PIL import Image as PILImage

from . import __version__
from .analysis import PLResult, PPResult
from .descriptors import (
    CLASS_CN, CLASS_EN, CLASS_ORDER, PLAssessment, PPAssessment,
    analyze_pl_binding, analyze_pp_binding,
)
from .quant import build_quant_figure
from .render import build_pl_figure, build_pp_figure
from .reportfigs import (
    build_bsa_figure, build_composition_figure, build_contact_figure,
    build_dashboard_figure, build_distance_stats_figure,
    build_enrichment_figure, build_fingerprint_figure,
    build_geometry_figure, build_hbond_geometry_figure,
    build_hotspot_figure, build_ligand_burial_map_figure,
    build_network_figure, build_pair_type_figure, build_type_class_figure,
)

PAGE_W, PAGE_H = A4
PAGE_W_L, PAGE_H_L = landscape(A4)
MARGIN_L = 18 * mm
MARGIN_R = 18 * mm
MARGIN_T = 22 * mm
MARGIN_B = 18 * mm

PRIMARY = colors.HexColor("#174534")
ACCENT = colors.HexColor("#2E7D5A")
GOLD = colors.HexColor("#C8A24A")
TEXT = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5B6570")
SOFT = colors.HexColor("#F4F8F5")
BORDER = colors.HexColor("#DCE6DF")

FONT_CN = "YaHei"
FONT_CNB = "YaHei-Bold"
FONT_EN = "Arial"
FONT_ENB = "Arial-Bold"
FONT_SERIF = "Georgia"
FONT_SERIFB = "Georgia-Bold"

TYPE_CN = {
    "hbond": "氢键", "salt": "盐桥", "pi": "π–π 堆积",
    "metal": "配位键", "hydrophobic": "疏水接触",
}
TYPE_EN = {
    "hbond": "H-bond", "salt": "Salt bridge", "pi": "π–π",
    "metal": "Metal", "hydrophobic": "Hydrophobic",
}


def _register_fonts():
    global FONT_CN, FONT_CNB, FONT_EN, FONT_ENB, FONT_SERIF, FONT_SERIFB
    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")

    def reg(name, path, index=0):
        full = os.path.join(fonts_dir, path)
        if os.path.exists(full):
            pdfmetrics.registerFont(TTFont(name, full, subfontIndex=index))
            return True
        return False

    cjk_ok = reg("YaHei", "msyh.ttc") and reg("YaHei-Bold", "msyhbd.ttc")
    if not cjk_ok:
        cjk_ok = reg("YaHei", "simsun.ttc") and reg("YaHei-Bold", "simhei.ttf")
    if cjk_ok:
        pdfmetrics.registerFontFamily("YaHei", normal="YaHei", bold="YaHei-Bold",
                                      italic="YaHei", boldItalic="YaHei-Bold")
    else:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        FONT_CN = FONT_CNB = "STSong-Light"

    if reg("Arial", "arial.ttf") and reg("Arial-Bold", "arialbd.ttf"):
        pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold",
                                      italic="Arial", boldItalic="Arial-Bold")
    else:
        FONT_EN = "Helvetica"
        FONT_ENB = "Helvetica-Bold"

    if reg("Georgia", "georgia.ttf") and reg("Georgia-Bold", "georgiab.ttf"):
        pdfmetrics.registerFontFamily("Georgia", normal="Georgia",
                                      bold="Georgia-Bold", italic="Georgia",
                                      boldItalic="Georgia-Bold")
    else:
        FONT_SERIF = "Times-Roman"
        FONT_SERIFB = "Times-Bold"


def _styles() -> dict[str, ParagraphStyle]:
    return {
        "h1": ParagraphStyle("h1", fontName=FONT_CNB, fontSize=16.0, leading=21,
                             textColor=PRIMARY, spaceAfter=3),
        "h1en": ParagraphStyle("h1en", fontName=FONT_EN, fontSize=9.0, leading=11,
                               textColor=MUTED, spaceAfter=4),
        "h2": ParagraphStyle("h2", fontName=FONT_CNB, fontSize=12.0, leading=15.5,
                             textColor=PRIMARY, spaceBefore=6, spaceAfter=3),
        "body_cn": ParagraphStyle("body_cn", fontName=FONT_CN, fontSize=10.2,
                                  leading=16.5, textColor=TEXT, alignment=TA_JUSTIFY,
                                  wordWrap="CJK", spaceAfter=6),
        "body_en": ParagraphStyle("body_en", fontName=FONT_EN, fontSize=9.2,
                                  leading=13.6, textColor=colors.HexColor("#4A5A50"),
                                  alignment=TA_JUSTIFY, spaceAfter=9),
        "bullet_cn": ParagraphStyle("bullet_cn", fontName=FONT_CN, fontSize=9.8,
                                    leading=15.5, textColor=TEXT, wordWrap="CJK",
                                    leftIndent=12, bulletIndent=0, spaceAfter=2),
        "bullet_en": ParagraphStyle("bullet_en", fontName=FONT_EN, fontSize=8.9,
                                    leading=12.8, textColor=MUTED, leftIndent=12,
                                    spaceAfter=8),
        "caption_cn": ParagraphStyle("caption_cn", fontName=FONT_CN, fontSize=9.2,
                                     leading=13.2, textColor=TEXT, wordWrap="CJK",
                                     spaceBefore=4, spaceAfter=3),
        "caption_en": ParagraphStyle("caption_en", fontName=FONT_EN, fontSize=8.6,
                                     leading=11.8, textColor=MUTED, spaceAfter=4),
        "interp_cn": ParagraphStyle("interp_cn", fontName=FONT_CN, fontSize=8.8,
                                    leading=13.0, textColor=colors.HexColor("#3F5147"),
                                    wordWrap="CJK", spaceAfter=2),
        "interp_en": ParagraphStyle("interp_en", fontName=FONT_EN, fontSize=8.1,
                                    leading=11.2, textColor=MUTED, spaceAfter=6),
        "table_cell": ParagraphStyle("table_cell", fontName=FONT_CN, fontSize=7.6,
                                     leading=9.8, textColor=TEXT, wordWrap="CJK"),
        "table_head": ParagraphStyle("table_head", fontName=FONT_CNB, fontSize=7.6,
                                     leading=9.8, textColor=colors.white,
                                     wordWrap="CJK"),
        "table_cell_c": ParagraphStyle("table_cell_c", fontName=FONT_CN, fontSize=7.6,
                                       leading=9.8, textColor=TEXT, alignment=TA_CENTER),
        "ref": ParagraphStyle("ref", fontName=FONT_EN, fontSize=8.0, leading=11.4,
                              textColor=MUTED, spaceAfter=4),
        "metric": ParagraphStyle("metric", fontName=FONT_CN, fontSize=6.8, leading=9,
                                 textColor=MUTED, alignment=TA_CENTER),
        "metric_num": ParagraphStyle("metric_num", fontName=FONT_SERIFB, fontSize=19,
                                     leading=22, alignment=TA_CENTER),
        "metric_label": ParagraphStyle("metric_label", fontName=FONT_CN, fontSize=7.8,
                                       leading=11, alignment=TA_CENTER,
                                       textColor=MUTED),
    }


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _cell(text, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def _fit_image(path: str, max_w: float, max_h: float) -> RLImage:
    iw, ih = PILImage.open(path).size
    ratio = min(max_w / iw, max_h / ih)
    return RLImage(path, width=iw * ratio, height=ih * ratio)


class InteractionReport:
    def __init__(self, result: PLResult | PPResult, output_path: str,
                 progress=None):
        self.result = result
        self.output_path = output_path
        self.progress = progress or (lambda msg: None)
        _register_fonts()
        self.styles = _styles()
        self.assessment: PLAssessment | PPAssessment | None = None
        self.fig_paths: dict[str, str] = {}
        self.tmpdir = tempfile.TemporaryDirectory(prefix="ligplotcn_report_")
        self.date = datetime.date.today().strftime("%Y-%m-%d")
        self.is_pl = isinstance(result, PLResult)

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------
    def _counts(self) -> dict[str, int]:
        counts = {"hbond": 0, "salt": 0, "pi": 0, "metal": 0, "hydrophobic": 0}
        for it in self.result.interactions:
            counts[it.kind] += 1
        return counts

    def _entity(self) -> str:
        if self.is_pl:
            lig = self.result.ligand
            return f"{lig.resname} {lig.resseq}{lig.icode.strip()} (Chain {lig.chain.strip() or '-'})"
        return f"Chains {self.result.chain_a.strip()} / {self.result.chain_b.strip()}"

    def _title(self) -> tuple[str, str, str]:
        if self.is_pl:
            return ("Ligand–Protein Interaction Atlas",
                    "配体–蛋白相互作用图谱", "Ligand–Protein Binding Report")
        return ("Protein–Protein Interface Atlas",
                "蛋白–蛋白界面图谱", "Protein–Protein Interface Report")

    def _metrics(self) -> list[tuple[str, str, colors.Color]]:
        s = self.result.stats
        a = self.assessment
        if self.is_pl:
            return [
                ("结合残基", str(len(self.result.nodes)), ACCENT),
                ("相互作用", str(len(self.result.interactions)), colors.HexColor("#3D6D96")),
                ("氢键", str(s["hbond"]), colors.HexColor("#2E7D32")),
                ("疏水接触", str(s["hydrophobic"]), colors.HexColor("#AE3B2C")),
                ("BSA 半和 (Å²)", f"{a.bsa:.0f}", colors.HexColor("#2E7D5A")),
                ("配体埋藏", f"{100 * a.descriptors.burial_fraction:.0f}%",
                 colors.HexColor("#C0604A")),
            ]
        interface = len(self.result.interface_a) + len(self.result.interface_b)
        return [
            ("界面残基", str(interface), ACCENT),
            ("相互作用", str(len(self.result.interactions)), colors.HexColor("#3D6D96")),
            ("氢键", str(s["hbond"]), colors.HexColor("#2E7D32")),
            ("盐桥", str(s["salt"]), colors.HexColor("#7B1FA2")),
            ("总埋藏 (Å²)", f"{a.total_buried:.0f}", colors.HexColor("#2E7D5A")),
            ("BSA 半和 (Å²)", f"{a.bsa:.0f}", colors.HexColor("#3F7A4F")),
        ]

    def _hotspot_labels(self, n: int = 3) -> list[str]:
        return [h["label"] for h in self.assessment.hotspots[:n]]

    def _dominant_type(self) -> str:
        counts = self._counts()
        return max(counts, key=lambda k: counts.get(k, 0))

    def _summary_cn(self) -> str:
        s = self.result.stats
        a = self.assessment
        if self.is_pl:
            lig = self.result.ligand
            return (
                f"本报告基于 PDB {self.result.structure.pdb_id} 中的配体 "
                f"{lig.resname} {lig.resseq}（链 {lig.chain.strip() or '-'}）。"
                f"共鉴定出 {len(self.result.nodes)} 个结合口袋残基和 "
                f"{len(self.result.interactions)} 条相互作用记录，其中氢键 "
                f"{s['hbond']} 个、盐桥 {s['salt']} 个、π–π 堆积 {s['pi']} 个、"
                f"金属配位 {s['metal']} 个、疏水接触 {s['hydrophobic']} 个。"
                f"配体自由态 SASA 为 {a.descriptors.sasa_free:.1f} Å²，结合后埋藏 "
                f"{a.descriptors.buried_area:.1f} Å²（{100 * a.descriptors.burial_fraction:.1f}%）；"
                f"结合埋藏面积（BSA，半和）为 {a.bsa:.1f} Å²，总埋藏面积为 "
                f"{a.total_buried:.1f} Å²。相互作用最频繁的残基为 "
                f"{'、'.join(self._hotspot_labels())}。"
            )
        return (
            f"本报告分析 PDB {self.result.structure.pdb_id} 中链 "
            f"{self.result.chain_a.strip()} 与链 {self.result.chain_b.strip()} 的蛋白–蛋白界面。"
            f"界面包含 {len(self.result.interface_a)} + {len(self.result.interface_b)} 个残基，"
            f"共 {len(self.result.interactions)} 条相互作用记录（氢键 {s['hbond']}、"
            f"盐桥 {s['salt']}、π–π {s['pi']}、疏水 {s['hydrophobic']}）。"
            f"总埋藏面积为 {a.total_buried:.1f} Å²，界面面积（BSA，半和）为 "
            f"{a.bsa:.1f} Å²，按经验分类属于"
            f"{ {'small': '小', 'medium': '中等', 'large': '大'}.get(a.interface_size_class, '—') }"
            f"界面。相互作用最密集的残基为 {'、'.join(self._hotspot_labels())}。"
        )

    def _summary_en(self) -> str:
        s = self.result.stats
        a = self.assessment
        if self.is_pl:
            lig = self.result.ligand
            return (
                f"This report analyses ligand {lig.resname} {lig.resseq} in PDB "
                f"{self.result.structure.pdb_id}. A total of {len(self.result.nodes)} "
                f"pocket residues and {len(self.result.interactions)} interaction records "
                f"were identified ({s['hbond']} hydrogen bonds, {s['salt']} salt bridges, "
                f"{s['pi']} π–π stacking, {s['metal']} metal coordination, and "
                f"{s['hydrophobic']} hydrophobic contacts). The ligand buries "
                f"{100 * a.descriptors.burial_fraction:.1f}% of its free SASA "
                f"({a.descriptors.buried_area:.1f} of {a.descriptors.sasa_free:.1f} Å²), "
                f"with a half-sum BSA of {a.bsa:.1f} Å² and a total buried surface of "
                f"{a.total_buried:.1f} Å². The most frequently engaged residues are "
                f"{', '.join(self._hotspot_labels())}."
            )
        return (
            f"This report analyses the interface between chains "
            f"{self.result.chain_a.strip()} and {self.result.chain_b.strip()} in PDB "
            f"{self.result.structure.pdb_id}. The interface comprises "
            f"{len(self.result.interface_a)} + {len(self.result.interface_b)} residues and "
            f"{len(self.result.interactions)} interaction records ({s['hbond']} hydrogen bonds, "
            f"{s['salt']} salt bridges, {s['pi']} π–π and {s['hydrophobic']} hydrophobic contacts). "
            f"The total buried surface is {a.total_buried:.1f} Å², with a half-sum BSA of "
            f"{a.bsa:.1f} Å², corresponding to a "
            f"{a.interface_size_class} interface by an empirical size classification. "
            f"The most interaction-dense residues are {', '.join(self._hotspot_labels())}."
        )

    def _class_summary(self, comp, lang: str = "cn") -> str:
        max_count = max(comp.counts.values()) if comp.counts else 0
        top = [c for c in CLASS_ORDER if comp.counts.get(c, 0) == max_count and max_count > 0]
        if not top:
            return "—"
        pct = comp.percentages.get(top[0], 0.0)
        if len(top) == 1:
            if lang == "cn":
                return f"以 {CLASS_CN[top[0]]} 类残基为主（{pct:.0f}%）"
            return f"dominated by {CLASS_EN[top[0]]} residues ({pct:.0f}%)"
        names_cn = "、".join(CLASS_CN[c] for c in top)
        names_en = ", ".join(CLASS_EN[c] for c in top)
        if lang == "cn":
            return f"残基类别较均衡（{names_cn} 各占 {pct:.0f}%）"
        return f"a balanced class distribution ({names_en}, each {pct:.0f}%)"

    def _findings(self) -> list[tuple[str, str]]:
        a = self.assessment
        s = self.result.stats
        dominant = self._dominant_type()
        out = []
        out.append((
            f"主要作用类型为{TYPE_CN[dominant]}，共 {self._counts()[dominant]} 条记录；"
            f"氢键与疏水作用的组合体现了界面的极性/非极性平衡。",
            f"The dominant interaction type is {TYPE_EN[dominant]} "
            f"({self._counts()[dominant]} records), indicating a balance between polar "
            f"and non-polar contributions.",
        ))
        if self.is_pl:
            out.append((
                f"配体埋藏比例 {100 * a.descriptors.burial_fraction:.1f}%，"
                f"结合埋藏面积（半和）{a.bsa:.1f} Å²；"
                f"每 100 Å² 界面约有 {a.interaction_density:.2f} 条相互作用记录。",
                f"The ligand buries {100 * a.descriptors.burial_fraction:.1f}% of its surface, "
                f"with a half-sum BSA of {a.bsa:.1f} Å² and "
                f"{a.interaction_density:.2f} records per 100 Å² of interface.",
            ))
            out.append((
                f"结合口袋的残基类别特征：{self._class_summary(a.pocket_composition, 'cn')}；"
                f"热点残基为 {'、'.join(self._hotspot_labels())}。",
                f"Pocket residue classes: {self._class_summary(a.pocket_composition, 'en')}; "
                f"the top hotspots are {', '.join(self._hotspot_labels())}.",
            ))
        else:
            out.append((
                f"总埋藏面积 {a.total_buried:.1f} Å²，属于"
                f"{ {'small': '小', 'medium': '中等', 'large': '大'}.get(a.interface_size_class, '—') }"
                f"界面；两侧埋藏较均衡。",
                f"The total buried surface is {a.total_buried:.1f} Å², corresponding to a "
                f"{a.interface_size_class} interface with balanced burial on both sides.",
            ))
            out.append((
                f"界面残基组成：疏水 {a.composition_all.percentages['Hydrophobic']:.0f}%、"
                f"芳香 {a.composition_all.percentages['Aromatic']:.0f}%、"
                f"极性 {a.composition_all.percentages['Polar']:.0f}%、"
                f"带电 {a.composition_all.percentages['Positive'] + a.composition_all.percentages['Negative']:.0f}%。",
                f"Interface composition: {a.composition_all.percentages['Hydrophobic']:.0f}% hydrophobic, "
                f"{a.composition_all.percentages['Aromatic']:.0f}% aromatic, "
                f"{a.composition_all.percentages['Polar']:.0f}% polar, and "
                f"{a.composition_all.percentages['Positive'] + a.composition_all.percentages['Negative']:.0f}% charged.",
            ))
        return out

    def _assessment_paragraphs(self) -> list[tuple[str, str]]:
        a = self.assessment
        if self.is_pl:
            return [
                (
                    f"从描述性指标看，该配体通过 {len(self.result.interactions)} 条记录与口袋结合，"
                    f"其中氢键 {self.result.stats['hbond']} 个、盐桥 {self.result.stats['salt']} 个、"
                    f"疏水接触 {self.result.stats['hydrophobic']} 个。配体埋藏比例较高，"
                    f"说明其大部分疏水表面被包埋；口袋残基类别以极性、芳香和带电残基混合为特征。"
                    f"上述结果支持该配体具有较充分的口袋互补性，但不能据此推断结合亲和力或药效。",
                    "Descriptively, the ligand engages the pocket through "
                    f"{len(self.result.interactions)} records, including "
                    f"{self.result.stats['hbond']} hydrogen bonds, {self.result.stats['salt']} "
                    f"salt bridges and {self.result.stats['hydrophobic']} hydrophobic contacts. "
                    "The high ligand burial fraction indicates substantial hydrophobic surface "
                    "occlusion, while the pocket combines polar, aromatic and charged residues. "
                    "These observations support shape and chemical complementarity but do not "
                    "constitute an affinity or efficacy prediction.",
                )
            ]
        return [
            (
                f"该界面总埋藏面积 {a.total_buried:.1f} Å²（半和 {a.bsa:.1f} Å²），"
                f"属于经验分类中的"
                f"{ {'small': '小', 'medium': '中等', 'large': '大'}.get(a.interface_size_class, '—') }"
                f"界面。界面同时具有氢键、盐桥和疏水接触，"
                f"疏水接触占记录的 {100 * a.hydrophobic_fraction:.0f}%，"
                f"提示疏水效应与静电互补共同参与界面稳定。"
                f"Arg/Asp 等带电残基构成明显的盐桥网络，是值得优先关注的界面热点。",
                f"The interface buries {a.total_buried:.1f} Å² in total "
                f"(half-sum BSA {a.bsa:.1f} Å²), corresponding to a "
                f"{a.interface_size_class} interface. Hydrogen bonds, salt bridges and "
                f"hydrophobic contacts co-exist; hydrophobic contacts account for "
                f"{100 * a.hydrophobic_fraction:.0f}% of the records, suggesting combined "
                f"hydrophobic and electrostatic complementarity. Charged Arg/Asp residues form "
                f"a prominent salt-bridge network and represent priority interface hotspots.",
            )
        ]

    # ------------------------------------------------------------------
    # 图
    # ------------------------------------------------------------------
    def _make_figures(self):
        tmp = self.tmpdir.name
        steps = []
        if self.is_pl:
            steps.append(("main", lambda: build_pl_figure(self.result, scale=1.0,
                                                          show_distances=True)))
        else:
            steps.append(("main", lambda: build_pp_figure(self.result, scale=1.0,
                                                          show_distances=True)))
        steps += [
            ("quant", lambda: build_quant_figure(self.result, scale=1.0)),
            ("fingerprint", lambda: build_fingerprint_figure(self.result, scale=1.0)),
            ("composition", lambda: build_composition_figure(self.result, self.assessment,
                                                             scale=1.0)),
            ("bsa", lambda: build_bsa_figure(self.result, self.assessment, scale=1.0)),
            ("geometry", lambda: build_geometry_figure(self.result, scale=1.0)),
            ("contact", lambda: build_contact_figure(self.result, self.assessment,
                                                     scale=1.0)),
            ("hotspot", lambda: build_hotspot_figure(self.result, self.assessment,
                                                     scale=1.0)),
            ("enrichment", lambda: build_enrichment_figure(self.result, self.assessment,
                                                           scale=1.0)),
            ("network", lambda: build_network_figure(self.result, scale=1.0)),
            ("typeclass", lambda: build_type_class_figure(self.result, self.assessment,
                                                          scale=1.0)),
            ("hbondjoint", lambda: build_hbond_geometry_figure(self.result, scale=1.0)),
            ("diststats", lambda: build_distance_stats_figure(self.result, scale=1.0)),
        ]
        if self.is_pl:
            steps.append(("ligandmap", lambda: build_ligand_burial_map_figure(
                self.result, self.assessment, scale=1.0)))
        else:
            steps.append(("pairtype", lambda: build_pair_type_figure(
                self.result, scale=1.0)))
        steps.append(("dashboard", lambda: build_dashboard_figure(
            self.result, self.assessment, scale=1.0)))
        for idx, (key, builder) in enumerate(steps, start=1):
            self.progress(f"生成图 {idx}/{len(steps)}：{key}")
            fig = builder()
            path = os.path.join(tmp, f"{key}.png")
            fig.savefig(path, dpi=300, facecolor="white",
                        bbox_inches="tight", pad_inches=0.12)
            self.fig_paths[key] = path
            fig.clear()

    def _figure_specs(self) -> list[dict]:
        s = self.result.stats
        a = self.assessment
        if self.is_pl:
            lig = self.result.ligand
            figure_defs = [
                ("main", "蛋白–配体相互作用二维示意图",
                 f"配体 {lig.resname} {lig.resseq} 位于中心，共 {len(self.result.nodes)} 个结合口袋残基；"
                 f"绿色虚线为氢键，紫色为盐桥，棕色为金属配位，红色睫毛弧为疏水接触。"
                 f"节点右上角数字为该残基的相互作用记录数。",
                 "Two-dimensional interaction diagram. The ligand is centered and surrounded "
                 f"by {len(self.result.nodes)} pocket residues. Green dashed lines denote hydrogen "
                 "bonds, purple dashed lines salt bridges, brown dashed lines metal coordination, "
                 "and red spoked arcs hydrophobic contacts. The number at each node is the "
                 "interaction-record count.",
                 f"该图给出了结合口袋的整体拓扑；{self._hotspot_labels(3)} 是视觉上最突出的接触残基。",
                 f"The diagram summarizes the pocket topology; "
                 f"{', '.join(self._hotspot_labels(3))} are the most visually prominent contact residues."),
                ("quant", "残基互作谱与距离分布",
                 "左图按残基统计相互作用记录数并按类型堆叠；右图为各类相互作用的距离分布及中位数，"
                 "用于比较氢键、盐桥、π–π、配位与疏水接触的空间尺度。",
                 "Left: per-residue interaction counts stacked by type. Right: distance "
                 "distributions with medians for each interaction class, allowing comparison of "
                 "the spatial scales of hydrogen bonds, salt bridges, π–π, metal coordination and "
                 "hydrophobic contacts.",
                 f"共 {len(self.result.interactions)} 条记录；记录数最多的残基为 "
                 f"{'、'.join(self._hotspot_labels(5))}。",
                 f"{len(self.result.interactions)} records in total; the most engaged residues are "
                 f"{', '.join(self._hotspot_labels(5))}."),
                ("fingerprint", "残基 × 作用类型指纹",
                 "热图每一行对应一个残基，每一列对应一种作用类型；颜色深浅与数字表示记录数。"
                 "右侧为每个残基的记录总数。",
                 "Heatmap of residues (rows) by interaction type (columns); colour intensity and "
                 "numbers indicate the record count. The right panel shows the total record count "
                 "per residue.",
                 "指纹图便于快速识别“多类型协同”的残基以及单一类型接触。",
                 "The fingerprint enables rapid identification of residues that combine multiple "
                 "interaction types versus those with a single dominant type."),
                ("composition", "结合口袋组成与配体属性",
                 "A：结合口袋与全部蛋白背景的残基类别组成比较；B：配体观测分子式、质量、环数、"
                 "可旋转键、供体/受体、形式电荷、埋藏比例与 BSA。",
                 "A: residue-class composition of the pocket versus the whole-protein background. "
                 "B: ligand formula, observed mass, ring count, rotatable bonds, donor/acceptor "
                 "counts, formal charge, burial fraction and BSA.",
                 f"口袋残基类别：{self._class_summary(a.pocket_composition, 'cn')}；"
                 f"配体埋藏比例为 {100 * a.descriptors.burial_fraction:.1f}%。",
                 f"Pocket residue classes: {self._class_summary(a.pocket_composition, 'en')}; "
                 f"ligand burial is {100 * a.descriptors.burial_fraction:.1f}%."),
                ("bsa", "埋藏表面积（BSA）分析",
                 "A：每个残基对结合埋藏的贡献（Å²）；B：半和 BSA、总埋藏面积、蛋白/配体各自埋藏面积与"
                 "配体埋藏比例。SASA 采用 Shrake–Rupley 方法、1.4 Å 探针。",
                 "A: per-residue contribution to the buried surface (Å²). B: half-sum BSA, total "
                 "buried area, protein/ligand buried contributions and ligand burial fraction. "
                 "SASA was computed with the Shrake–Rupley method and a 1.4 Å probe.",
                 f"半和 BSA 为 {a.bsa:.1f} Å²，总埋藏 {a.total_buried:.1f} Å²；"
                 f"配体埋藏 {a.descriptors.buried_area:.1f} Å²。",
                 f"The half-sum BSA is {a.bsa:.1f} Å² and the total buried surface is "
                 f"{a.total_buried:.1f} Å²; the ligand buries {a.descriptors.buried_area:.1f} Å²."),
                ("geometry", "相互作用几何与疏水接触统计",
                 "A：氢键供体–受体距离与 D–H···A 角度散点图，虚线为 3.5 Å 与 120° 参考线；"
                 "B：各残基的疏水原子对接触数与最小距离。",
                 "A: donor–acceptor distance versus D–H···A angle for hydrogen bonds, with 3.5 Å "
                 "and 120° reference lines. B: hydrophobic atom-pair contact counts and minimum "
                 "distances per residue.",
                 "该图用于检查氢键几何是否处于常见范围，并量化疏水接触的多重性。",
                 "This figure checks whether hydrogen-bond geometry falls within common ranges and "
                 "quantifies the multiplicity of hydrophobic contacts."),
                ("contact", "配体逐原子埋藏与元素组成",
                 "A：每个配体原子的埋藏 SASA 及埋藏比例，圆点标记参与极性相互作用的原子；"
                 "B：配体重原子的元素组成。",
                 "A: buried SASA and burial fraction for each ligand atom; dots mark atoms involved "
                 "in polar interactions. B: element composition of ligand heavy atoms.",
                 "逐原子埋藏图可定位深埋于口袋的疏水骨架与暴露的极性基团。",
                 "The per-atom burial profile locates deeply buried hydrophobic scaffolds and more "
                 "exposed polar groups."),
                ("hotspot", "残基热点景观",
                 "以每个残基的埋藏表面积（x 轴）对相互作用记录数（y 轴）作图；气泡面积表示疏水原子对接触数。"
                 "虚线为中位数，右上象限代表“高埋藏且高参与”的残基，可作为优先关注的热点。",
                 "Buried surface area per residue (x) versus interaction records (y); bubble area "
                 "encodes hydrophobic atom-pair contacts. Dashed lines mark medians, and the "
                 "upper-right quadrant identifies residues with both high burial and high engagement.",
                 "该图将“参与度”与“埋藏贡献”放在同一坐标系中，避免仅按接触数量误判热点；"
                 "右上象限残基通常最值得后续突变或结构验证。",
                 "By combining engagement and burial in one coordinate system, this view avoids "
                 "defining hotspots solely by contact counts. Residues in the upper-right quadrant "
                 "are generally the most attractive candidates for mutational or structural follow-up."),
                ("enrichment", "残基类别富集",
                 "比较结合口袋与全部蛋白背景的六类残基比例，以百分点差值表示富集或缺失；"
                 "括号中给出“口袋比例 / 背景比例”和该类残基数。",
                 "Six residue classes compared between the pocket and the whole-protein background. "
                 "Bars show percentage-point differences; parentheses give pocket/background "
                 "percentages and class counts.",
                 "富集图用于判断口袋是否偏好芳香、疏水或带电残基；由于口袋残基数有限，"
                 "百分点差异应结合计数谨慎解读。",
                 "The enrichment view indicates whether the pocket favours aromatic, hydrophobic or "
                 "charged residues. Because pocket counts are limited, percentage-point differences "
                 "should be interpreted together with the underlying counts."),
                ("network", "相互作用网络",
                 "以环形网络展示配体–残基关系：节点大小表示相互作用记录数，边宽表示记录数，"
                 "颜色表示主导作用类型；配体位于中心，残基节点颜色表示残基类别。",
                 "Circular ligand–residue network. Node size and edge width encode interaction "
                 "records; edge colour indicates the dominant interaction type. The ligand is at the "
                 "centre, and residue-node colour indicates the residue class.",
                 "网络强调多价残基和残基–配体连接的整体拓扑，可与二维原子级示意图互补阅读。",
                 "The network emphasizes multivalent residues and the overall topology of "
                 "ligand–residue connections and complements the atom-level two-dimensional diagram."),
                ("typeclass", "作用类型 × 残基类别",
                 "热图每行为一种残基类别、每列为一种作用类型，单元格数字为涉及的相互作用记录数；"
                 "右侧条形为各类别总涉及数。该图把化学性质与作用方式对应起来。",
                 "Heatmap of residue classes (rows) by interaction type (columns). Cell numbers are "
                 "residue–interaction involvements and the right bars give class totals, linking "
                 "residue chemistry to interaction mode.",
                 f"口袋以 {self._class_summary(a.pocket_composition, 'cn')} 为特征；"
                 "可据此判断氢键、盐桥与疏水接触分别由哪类残基承担。",
                 f"The pocket is {self._class_summary(a.pocket_composition, 'en')}; the view shows "
                 "which residue classes carry hydrogen bonds, salt bridges and hydrophobic contacts."),
                ("hbondjoint", "氢键距离–角度联合分布",
                 "主图每个点为一个氢键的供体–受体重原子距离与 D–H···A 角度；顶部与右侧为边缘分布。"
                 "绿色阴影标出 2.4–3.5 Å 与 ≥120° 的常见有利区间，虚线为中位数。",
                 "Each point is a hydrogen bond plotted by donor–acceptor distance and D–H···A angle, "
                 "with marginal histograms on the top and right. Green shading marks the commonly "
                 "favourable 2.4–3.5 Å and ≥120° ranges; dashed lines are medians.",
                 "联合分布同时考虑距离与角度，避免仅凭距离判定氢键；落在阴影区的记录几何更典型，"
                 "偏离阴影区的记录应结合局部环境谨慎解读。",
                 "Jointly considering distance and angle avoids assigning hydrogen bonds by distance "
                 "alone. Records inside the shaded region have more typical geometry, whereas outliers "
                 "should be interpreted together with the local environment."),
                ("diststats", "相互作用距离统计",
                 "A：五类相互作用距离的箱线与逐记录散点，箱体为四分位距、粗线为中位数、须为范围；"
                 "B：累积分布，可读取任一类在给定距离阈值内包含的记录比例。",
                 "A: box-and-scatter summary of distances for the five interaction classes; boxes span "
                 "the interquartile range, thick lines mark medians and whiskers span the full range. "
                 "B: cumulative distributions giving the fraction of records within any distance cutoff.",
                 "距离统计给出各类作用的尺度与离散度；累积曲线便于在统一阈值下比较不同作用的包含比例。",
                 "The statistics summarize the scale and spread of each interaction class, and the "
                 "cumulative curves allow the fraction of records within a common cutoff to be compared."),
                ("ligandmap", "配体逐原子埋藏空间图",
                 "将配体重原子投影到二维平面，颜色表示每个原子的埋藏比例（0–1，浅绿至深绿），"
                 "金色描边标记参与极性相互作用的杂原子。",
                 "Ligand heavy atoms projected onto two dimensions; colour encodes the per-atom burial "
                 "fraction (0–1, light to dark green) and gold outlines mark heteroatoms engaged in "
                 "polar interactions.",
                 "空间图显示埋藏沿配体骨架的分布，可定位深埋的疏水核心与相对暴露的极性端，"
                 "并与逐原子埋藏剖面图（图 7）互为补充。",
                 "The map shows how burial is distributed along the ligand scaffold, locating deeply "
                 "buried hydrophobic cores and more exposed polar ends, and complements the per-atom "
                 "burial profile in Figure 7."),
                ("dashboard", "综合指标仪表盘",
                 "汇总 BSA、配体埋藏、氢键、疏水、口袋残基数与相互作用密度，并以环形图展示作用类型组成、"
                 "以条形图展示热点残基。",
                 "Summary dashboard of BSA, ligand burial, hydrogen bonds, hydrophobic contacts, "
                 "pocket size and interaction density, with a donut chart of interaction types and "
                 "a bar chart of hotspot residues.",
                 "仪表盘用于快速比较不同结构/配体的结合特征，适合作为汇报首页图。",
                 "The dashboard enables rapid comparison of binding profiles across structures or "
                 "ligands and is suitable as an opening summary figure."),
            ]
        else:
            figure_defs = [
                ("main", "蛋白–蛋白界面二维示意图",
                 f"链 {self.result.chain_a.strip()} 与链 {self.result.chain_b.strip()} 的界面残基分列左右两栏；"
                 "绿色虚线为氢键，紫色为盐桥，棕色为金属配位，橙红点线为疏水接触。"
                 "节点右上角数字为该残基的相互作用记录数。",
                 "Two-dimensional interface diagram. Interface residues of chains "
                 f"{self.result.chain_a.strip()} and {self.result.chain_b.strip()} are arranged in "
                 "two columns. Green dashed lines denote hydrogen bonds, purple dashed lines salt "
                 "bridges, brown dashed lines metal coordination, and orange dotted lines "
                 "hydrophobic contacts. Node numbers give interaction-record counts.",
                 f"界面由 {len(self.result.interface_a)} + {len(self.result.interface_b)} 个残基组成，"
                 f"热点为 {'、'.join(self._hotspot_labels(3))}。",
                 f"The interface comprises {len(self.result.interface_a)} + "
                 f"{len(self.result.interface_b)} residues; hotspots include "
                 f"{', '.join(self._hotspot_labels(3))}."),
                ("quant", "残基互作谱与距离分布",
                 "左图按残基统计相互作用记录数并按类型堆叠；右图为五类相互作用的距离分布及中位数。",
                 "Left: per-residue interaction records stacked by type. Right: distance "
                 "distributions with medians for the five interaction classes.",
                 f"共 {len(self.result.interactions)} 条记录；记录数最多的残基为 "
                 f"{'、'.join(self._hotspot_labels(5))}。",
                 f"{len(self.result.interactions)} records in total; the most engaged residues are "
                 f"{', '.join(self._hotspot_labels(5))}."),
                ("fingerprint", "残基 × 作用类型指纹",
                 "热图展示每个界面残基的五类相互作用记录数，右侧为总量。",
                 "Heatmap of the five interaction classes for each interface residue, with totals "
                 "on the right.",
                 "可快速识别同时参与氢键、盐桥和疏水接触的多价残基。",
                 "This view highlights multivalent residues that simultaneously contribute "
                 "hydrogen bonds, salt bridges and hydrophobic contacts."),
                ("composition", "界面残基组成与埋藏平衡",
                 "A：界面与全部蛋白背景的残基类别组成比较；B：两侧链的埋藏面积、总埋藏、BSA 与界面大小分类。",
                 "A: residue-class composition of the interface versus the protein background. "
                 "B: buried area per chain, total buried surface, half-sum BSA and interface-size "
                 "classification.",
                 f"界面总埋藏 {a.total_buried:.1f} Å²，半和 BSA {a.bsa:.1f} Å²，"
                 f"属于 {a.interface_size_class} 界面。",
                 f"Total buried surface {a.total_buried:.1f} Å²; half-sum BSA {a.bsa:.1f} Å²; "
                 f"{a.interface_size_class} interface."),
                ("bsa", "界面埋藏表面积（BSA）分析",
                 "A：每个界面残基的埋藏贡献；B：两侧链的埋藏面积平衡、总埋藏与界面大小分类。"
                 "SASA 采用 Shrake–Rupley、1.4 Å 探针。",
                 "A: per-residue buried-surface contribution. B: balance of burial between the two "
                 "chains, total buried surface and interface-size classification. SASA used "
                 "Shrake–Rupley with a 1.4 Å probe.",
                 "两侧埋藏较均衡，说明界面为双向互补而非单侧插入。",
                 "The balanced burial on both sides indicates a two-sided complementary interface "
                 "rather than a one-sided insertion."),
                ("geometry", "相互作用几何与疏水接触统计",
                 "A：氢键距离–角度散点图；B：各残基疏水原子对接触数与最小距离。",
                 "A: hydrogen-bond distance–angle scatter. B: hydrophobic atom-pair contacts and "
                 "minimum distances per residue.",
                 "用于检查界面氢键几何与疏水接触的多重性。",
                 "The figure examines hydrogen-bond geometry and the multiplicity of hydrophobic "
                 "contacts across the interface."),
                ("contact", "残基–残基接触矩阵",
                 "气泡图的行/列为两条链的界面残基；气泡大小和数字表示相互作用记录数，颜色表示主导作用类型。",
                 "Bubble matrix of interface residues from the two chains. Bubble size and labels "
                 "indicate interaction records; colour indicates the dominant interaction type.",
                 f"最密集的残基对涉及 {'、'.join(self._hotspot_labels(3))} 等热点。",
                 f"The densest residue pairs involve hotspots such as "
                 f"{', '.join(self._hotspot_labels(3))}."),
                ("hotspot", "界面热点景观",
                 "以每个界面残基的埋藏表面积（x 轴）对相互作用记录数（y 轴）作图；气泡面积表示疏水原子对接触数，"
                 "颜色区分两条链；虚线为中位数，右上象限为“高埋藏且高参与”的候选热点。",
                 "Buried surface area per interface residue (x) versus interaction records (y). "
                 "Bubble area encodes hydrophobic atom-pair contacts and colour distinguishes the "
                 "two chains. Dashed lines mark medians; the upper-right quadrant highlights "
                 "candidate hotspots with high burial and high engagement.",
                 "该图综合了接触频次与埋藏贡献，可避免仅凭接触数量或将埋藏面积单独作为热点判据。",
                 "This view integrates contact frequency and burial contribution, avoiding the use "
                 "of either contact counts or buried area alone as a hotspot criterion."),
                ("enrichment", "界面残基类别富集",
                 "比较界面与全部蛋白背景的六类残基比例，以百分点差值表示富集或缺失；"
                 "括号中给出“界面比例 / 背景比例”和该类残基数。",
                 "Six residue classes compared between the interface and the whole-protein "
                 "background. Bars show percentage-point differences; parentheses give "
                 "interface/background percentages and class counts.",
                 "该图揭示界面是否在芳香、疏水或带电残基上具有偏好，是解读界面物理化学互补性的补充证据。",
                 "The view reveals whether the interface preferentially uses aromatic, hydrophobic "
                 "or charged residues, providing complementary evidence on physicochemical "
                 "complementarity."),
                ("network", "蛋白–蛋白相互作用网络",
                 "环形网络展示两条链的界面残基及其相互作用；节点大小表示记录数，边宽表示记录数，"
                 "颜色表示主导作用类型，节点颜色区分链 A/链 B。",
                 "Circular network of interface residues from the two chains. Node size and edge "
                 "width encode interaction records; edge colour indicates the dominant interaction "
                 "type, and node colour distinguishes chains A and B.",
                 "网络图突出多价残基与残基对连接模式，便于识别界面中的枢纽残基和盐桥网络。",
                 "The network highlights multivalent residues and residue-pair connectivity, "
                 "facilitating identification of hub residues and salt-bridge networks."),
                ("typeclass", "作用类型 × 残基类别",
                 "热图每行为一种残基类别、每列为一种作用类型，单元格数字为涉及的记录数；"
                 "右侧条形为各类别总涉及数。该图揭示界面化学性质与作用方式的对应关系。",
                 "Heatmap of residue classes (rows) by interaction type (columns). Cell numbers are "
                 "residue–interaction involvements and the right bars give class totals, revealing how "
                 "interface chemistry maps onto interaction modes.",
                 f"界面以 {self._class_summary(a.composition_all, 'cn')} 为特征；"
                 "该图显示盐桥、氢键与疏水接触分别由哪类残基承担。",
                 f"The interface is {self._class_summary(a.composition_all, 'en')}; the view shows "
                 "which residue classes carry salt bridges, hydrogen bonds and hydrophobic contacts."),
                ("hbondjoint", "氢键距离–角度联合分布",
                 "主图每个点为一个界面氢键的距离与 D–H···A 角度，顶部与右侧为边缘分布；"
                 "绿色阴影为 2.4–3.5 Å 与 ≥120° 的常见有利区间，虚线为中位数。",
                 "Each point is an interface hydrogen bond plotted by donor–acceptor distance and "
                 "D–H···A angle, with marginal histograms. Green shading marks the commonly favourable "
                 "2.4–3.5 Å and ≥120° ranges; dashed lines are medians.",
                 "该图用于检查界面氢键是否具有典型几何；偏离有利区间的记录可能反映较弱的极性接触或几何张力。",
                 "The figure checks whether interface hydrogen bonds adopt typical geometry; records "
                 "outside the favourable region may indicate weaker polar contacts or geometric strain."),
                ("diststats", "相互作用距离统计",
                 "A：五类作用距离的箱线与逐记录散点；B：累积分布。箱体为四分位距、粗线为中位数、须为范围。",
                 "A: box-and-scatter summaries of distances for the five interaction classes. "
                 "B: cumulative distributions. Boxes span the interquartile range, thick lines mark "
                 "medians and whiskers span the full range.",
                 "该图给出界面各类作用的距离尺度与离散度，并可在统一阈值下比较包含比例。",
                 "The figure reports the distance scale and spread of each interface interaction "
                 "class and enables comparison of coverage at a common cutoff."),
                ("pairtype", "残基对作用组成",
                 "行为记录数最高的 12 个界面残基对，列为作用类型；单元格数字为记录数，"
                 "右侧条形为残基对总记录数。",
                 "The 12 interface residue pairs with the most records (rows) by interaction type "
                 "(columns). Cell numbers are record counts and the right bars give pair totals.",
                 "该图揭示哪些残基对通过多类型作用形成稳定接触，可作为界面枢纽与突变设计的优先对象。",
                 "The view reveals which residue pairs form stable contacts through multiple "
                 "interaction types, prioritizing interface hubs for mutational design."),
                ("dashboard", "综合指标仪表盘",
                 "汇总 BSA、总埋藏、氢键、盐桥、界面残基数与作用密度，并展示作用类型组成与热点残基。",
                 "Summary dashboard of BSA, total buried surface, hydrogen bonds, salt bridges, "
                 "interface size and interaction density, with interaction composition and hotspot "
                 "residues.",
                 "适合用于快速比较多组复合物的界面特征。",
                 "Suitable for rapid comparison of interface features across complexes."),
            ]
        numbered = []
        landscape_keys = {
            "quant", "fingerprint", "composition", "bsa", "geometry",
            "contact", "hotspot", "enrichment", "typeclass", "diststats",
            "pairtype", "dashboard",
        }
        for idx, item in enumerate(figure_defs, start=1):
            key, cn_title, cn_caption, en_caption, icn, ien = item
            numbered.append({
                "key": key, "num": idx,
                "cn": f"{cn_title}。{cn_caption}",
                "en": en_caption,
                "icn": icn, "ien": ien,
                "landscape": key in landscape_keys,
            })
        return numbered

    # ------------------------------------------------------------------
    # PDF 排版
    # ------------------------------------------------------------------
    def _section(self, title_cn: str, title_en: str) -> list:
        return [
            Spacer(1, 3 * mm),
            _p(escape(title_cn), self.styles["h1"]),
            _p(escape(title_en), self.styles["h1en"]),
            HRFlowable(width="100%", thickness=0.8, color=BORDER,
                       spaceBefore=0, spaceAfter=5),
        ]

    def _metric_cards(self) -> Table:
        metrics = self._metrics()
        data = []
        row = []
        for label, value, color in metrics:
            hexcolor = "#%02X%02X%02X" % (
                int(color.red * 255), int(color.green * 255), int(color.blue * 255)
            )
            num = _p(f'<font color="{hexcolor}">{escape(value)}</font>',
                     self.styles["metric_num"])
            text = _p(escape(label), self.styles["metric_label"])
            card = Table([[num], [text]], colWidths=[50 * mm])
            card.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
            row.append(card)
            if len(row) == 3:
                data.append(row)
                row = []
        if row:
            data.append(row)
        table = Table(data, colWidths=[58 * mm] * 3, hAlign="CENTER")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), SOFT),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 4, colors.white),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        return table

    def _summary_story(self) -> list:
        story = self._section("摘要", "Executive Summary")
        story.append(_p(escape(self._summary_cn()), self.styles["body_cn"]))
        story.append(_p(escape(self._summary_en()), self.styles["body_en"]))
        story.append(Spacer(1, 2 * mm))
        story.append(self._metric_cards())
        story.append(Spacer(1, 5 * mm))

        story += self._section("主要发现", "Key Findings")
        for cn, en in self._findings():
            story.append(Paragraph(escape(cn), self.styles["bullet_cn"],
                                   bulletText="•"))
            story.append(_p(escape(en), self.styles["bullet_en"]))

        story += self._section("总体评价", "Overall Assessment")
        for cn, en in self._assessment_paragraphs():
            story.append(_p(escape(cn), self.styles["body_cn"]))
            story.append(_p(escape(en), self.styles["body_en"]))
        return story

    def _figure_extra(self, key: str) -> tuple[str, str]:
        if self.is_pl:
            notes = {
                "main": (
                    "图中仅标注参与相互作用的配体杂原子；蛋白原子名称、完整原子对与阈值见定量明细表。"
                    "节点右侧的数字为该残基的相互作用记录总数。",
                    "Only ligand heteroatoms engaged in interactions are labelled; full protein atom "
                    "names, atom pairs and thresholds are listed in the quantitative table. The number "
                    "next to each node is the total interaction-record count for that residue.",
                ),
                "quant": (
                    "左图按作用类型堆叠每个残基的记录数；右图每条记录为一个点，虚线为中位数。"
                    "疏水记录的距离为残基与配体之间的最小重原子距离。",
                    "Left: records per residue stacked by type. Right: each point is one record and "
                    "dashed lines are medians. For hydrophobic records, distance is the minimum "
                    "heavy-atom distance between the residue and the ligand.",
                ),
                "fingerprint": (
                    "单元格数字为记录数，颜色越深表示记录越多；右侧总量用于比较残基的多价性。"
                    "零记录以“·”表示。",
                    "Cell numbers are record counts and darker colour indicates more records; the "
                    "right bar shows total records per residue. Zero counts are shown as dots.",
                ),
                "composition": (
                    "A 图百分比以各自残基集合为分母；B 图质量为结构文件中实际建模原子的观测质量，"
                    "若氢未建模则为重原子观测质量。",
                    "In A, percentages are calculated within each residue set. In B, mass is the "
                    "observed mass of modelled atoms; if hydrogens are absent, it is a heavy-atom "
                    "observed mass.",
                ),
                "bsa": (
                    "半和 BSA 为两侧埋藏面积之和的一半；总埋藏面积为两侧埋藏之和。"
                    "SASA 采用 Shrake–Rupley、1.4 Å 探针和 96 个球面点。",
                    "Half-sum BSA is half of the sum of buried areas on both sides; total buried "
                    "surface is their sum. SASA used Shrake–Rupley with a 1.4 Å probe and 96 sphere "
                    "points.",
                ),
                "geometry": (
                    "每个氢键点对应一个供体–受体原子对，参考线为 3.5 Å 与 120°；"
                    "疏水柱状图标注原子对接触数与最小距离。",
                    "Each hydrogen-bond point is a donor–acceptor pair with reference lines at 3.5 Å "
                    "and 120°. Hydrophobic bars show atom-pair contacts and minimum distances.",
                ),
                "contact": (
                    "逐原子埋藏比例 = 埋藏 SASA / 自由原子 SASA；圆点标记参与极性相互作用的原子。"
                    "B 图给出配体重原子的元素组成。",
                    "Per-atom burial fraction = buried SASA / free-atom SASA; dots mark atoms in "
                    "polar interactions. Panel B shows the element composition of heavy atoms.",
                ),
                "hotspot": (
                    "x 轴为逐残基埋藏面积，y 轴为记录数；气泡面积为疏水原子对接触数；"
                    "右上象限代表高埋藏且高参与的残基。",
                    "x-axis: per-residue buried area; y-axis: interaction records; bubble area: "
                    "hydrophobic atom-pair contacts; the upper-right quadrant highlights residues "
                    "with both high burial and high engagement.",
                ),
                "enrichment": (
                    "差值为“口袋比例 − 蛋白背景比例”的百分点；括号内给出比例与残基数。"
                    "小样本计数下差异可能不稳定。",
                    "Bars show pocket-minus-background percentage points; parentheses give "
                    "percentages and counts. Differences can be unstable with small counts.",
                ),
                "network": (
                    "中心为配体，外环为残基；节点颜色表示残基类别，边色表示主导作用类型，"
                    "节点与边的尺寸均随记录数增加。",
                    "The ligand is at the centre and residues form the outer ring. Node colour "
                    "indicates residue class; edge colour indicates dominant interaction type. Node "
                    "and edge sizes increase with record count.",
                ),
                "typeclass": (
                    "行合计为残基类别在所有作用类型中的涉及次数；同一记录若同时涉及两类作用仅计入对应列。"
                    "小样本计数下个别单元格可能为零。",
                    "Row totals count involvements of a residue class across interaction types; each "
                    "record contributes to its own type column. Individual cells can be zero when the "
                    "underlying counts are small.",
                ),
                "hbondjoint": (
                    "阴影区为常用的有利距离–角度范围；点越靠近右上表示距离更短、角度更接近线性。"
                    "无显式氢时角度由虚拟氢估计。",
                    "Shaded regions mark commonly favourable distance–angle ranges; points towards the "
                    "upper right indicate shorter and more linear hydrogen bonds. Angles are estimated "
                    "with virtual hydrogens when explicit hydrogens are absent.",
                ),
                "diststats": (
                    "箱体为四分位距，粗线为中位数，水平须为最小–最大范围；散点为单条记录。"
                    "B 图在统一距离阈值下可直接读取累计比例。",
                    "Boxes span the interquartile range, thick lines are medians and horizontal "
                    "whiskers span the minimum–maximum range; points are individual records. Panel B "
                    "gives cumulative fractions at any common distance cutoff.",
                ),
                "ligandmap": (
                    "投影为配体主平面二维视图，颜色仅表示埋藏比例，不表示原子类型；"
                    "金色描边为参与极性相互作用的杂原子。",
                    "The projection is a two-dimensional view of the ligand plane; colour encodes "
                    "burial only, not atom type. Gold outlines mark heteroatoms engaged in polar "
                    "interactions.",
                ),
                "dashboard": (
                    "环形图记录数之和等于相互作用总记录数；热点条形图给出记录数与逐残基 BSA。"
                    "所有指标为描述性统计。",
                    "The sum of donut segments equals total interaction records; hotspot bars show "
                    "record counts and per-residue BSA. All metrics are descriptive.",
                ),
            }
        else:
            notes = {
                "main": (
                    "节点右侧数字为该残基的相互作用记录数；边色表示主导作用类型。"
                    "完整原子对、距离与角度见定量明细表。",
                    "The number next to each node is the interaction-record count; edge colour "
                    "indicates the dominant interaction type. Complete atom pairs, distances and "
                    "angles are listed in the quantitative table.",
                ),
                "quant": (
                    "左图按作用类型堆叠每个残基的记录数；右图每条记录为一个点，虚线为中位数。"
                    "疏水记录的距离为残基间最小重原子距离。",
                    "Left: records per residue stacked by type. Right: each point is one record and "
                    "dashed lines are medians. For hydrophobic records, distance is the minimum "
                    "heavy-atom distance between residues.",
                ),
                "fingerprint": (
                    "单元格数字为记录数，颜色越深表示记录越多；右侧总量用于比较残基的多价性。"
                    "零记录以“·”表示。",
                    "Cell numbers are record counts and darker colour indicates more records; the "
                    "right bar shows total records per residue. Zero counts are shown as dots.",
                ),
                "composition": (
                    "A 图百分比以各自残基集合为分母；B 图给出两侧链的埋藏面积、总埋藏、半和 BSA 与界面大小分类。",
                    "In A, percentages are calculated within each residue set. B shows buried area "
                    "per chain, total buried surface, half-sum BSA and interface-size class.",
                ),
                "bsa": (
                    "半和 BSA 为两侧埋藏面积之和的一半；总埋藏面积为两侧埋藏之和。"
                    "SASA 采用 Shrake–Rupley、1.4 Å 探针和 96 个球面点。",
                    "Half-sum BSA is half of the sum of buried areas on both sides; total buried "
                    "surface is their sum. SASA used Shrake–Rupley with a 1.4 Å probe and 96 sphere "
                    "points.",
                ),
                "geometry": (
                    "每个氢键点对应一个供体–受体原子对，参考线为 3.5 Å 与 120°；"
                    "疏水柱状图标注原子对接触数与最小距离。",
                    "Each hydrogen-bond point is a donor–acceptor pair with reference lines at 3.5 Å "
                    "and 120°. Hydrophobic bars show atom-pair contacts and minimum distances.",
                ),
                "contact": (
                    "矩阵行列分别为两条链的主要界面残基；气泡大小为记录数，颜色为主导作用类型，"
                    "数字为记录数。",
                    "Rows and columns are the main interface residues of the two chains; bubble size "
                    "and labels show record counts, and colour indicates the dominant interaction type.",
                ),
                "hotspot": (
                    "x 轴为逐残基埋藏面积，y 轴为记录数；气泡面积为疏水原子对接触数，"
                    "颜色区分链；右上象限为高埋藏且高参与的热点。",
                    "x-axis: per-residue buried area; y-axis: interaction records; bubble area: "
                    "hydrophobic atom-pair contacts; colour distinguishes chains. The upper-right "
                    "quadrant highlights hotspots with high burial and high engagement.",
                ),
                "enrichment": (
                    "差值为“界面比例 − 蛋白背景比例”的百分点；括号内给出比例与残基数。"
                    "小样本计数下差异可能不稳定。",
                    "Bars show interface-minus-background percentage points; parentheses give "
                    "percentages and counts. Differences can be unstable with small counts.",
                ),
                "network": (
                    "节点颜色区分链 A/链 B；节点大小与边宽表示记录数，边色表示主导作用类型。"
                    "该图突出多价残基与残基对连接模式。",
                    "Node colour distinguishes chains A and B; node size and edge width encode "
                    "records, and edge colour indicates the dominant type. The view highlights "
                    "multivalent residues and pair connectivity.",
                ),
                "typeclass": (
                    "行合计为残基类别在所有作用类型中的涉及次数；蛋白–蛋白模式下每条记录涉及两侧残基，"
                    "按各自残基类别分别计数。",
                    "Row totals count involvements of a residue class across interaction types. In "
                    "protein–protein mode each record involves residues on both sides, which are "
                    "counted under their own classes.",
                ),
                "hbondjoint": (
                    "阴影区为常用的有利距离–角度范围；无显式氢时角度由虚拟氢估计，"
                    "界面氢键可能包含较弱或非典型几何。",
                    "Shaded regions mark commonly favourable distance–angle ranges. Angles are "
                    "estimated with virtual hydrogens when explicit hydrogens are absent; interface "
                    "hydrogen bonds may include weaker or non-canonical geometry.",
                ),
                "diststats": (
                    "箱体为四分位距，粗线为中位数，水平须为最小–最大范围；散点为单条记录。"
                    "疏水记录的距离为残基间最小重原子距离。",
                    "Boxes span the interquartile range, thick lines are medians and whiskers span "
                    "the minimum–maximum range; points are individual records. For hydrophobic "
                    "records, distance is the minimum heavy-atom distance between residues.",
                ),
                "pairtype": (
                    "仅统计可归属到具体残基对的记录（如残基对合并的疏水/盐桥记录）；"
                    "行按总记录数降序，最多显示 12 对。",
                    "Only records assignable to a specific residue pair (e.g. pair-aggregated "
                    "hydrophobic or salt-bridge records) are counted. Rows are sorted by total records "
                    "and the top 12 pairs are shown.",
                ),
                "dashboard": (
                    "环形图记录数之和等于相互作用总记录数；热点条形图给出记录数与逐残基 BSA。"
                    "所有指标为描述性统计。",
                    "The sum of donut segments equals total interaction records; hotspot bars show "
                    "record counts and per-residue BSA. All metrics are descriptive.",
                ),
            }
        return notes.get(key, ("", ""))

    def _figure_guide_story(self) -> list:
        specs = self._figure_specs()
        guide = {
            "main": ("整体结合拓扑", "Overall binding topology"),
            "quant": ("逐残基作用强度与距离", "Per-residue engagement and distances"),
            "fingerprint": ("作用类型协同", "Interaction-type co-occurrence"),
            "composition": ("化学互补性", "Chemical complementarity"),
            "bsa": ("埋藏面积贡献", "Burial contribution"),
            "geometry": ("几何合理性与接触多重性", "Geometry and contact multiplicity"),
            "contact": ("残基配对 / 配体原子埋藏", "Residue pairing / ligand atom burial"),
            "hotspot": ("热点残基识别", "Hotspot identification"),
            "enrichment": ("残基类别偏好", "Residue-class preference"),
            "network": ("界面网络拓扑", "Interface network topology"),
            "typeclass": ("作用类型与残基化学", "Interaction type versus residue chemistry"),
            "hbondjoint": ("氢键几何合理性", "Hydrogen-bond geometry quality"),
            "diststats": ("距离尺度与离散度", "Distance scale and spread"),
            "ligandmap": ("埋藏空间分布", "Spatial burial pattern"),
            "pairtype": ("残基对作用组成", "Residue-pair interaction composition"),
            "dashboard": ("综合概览", "Integrated overview"),
        }
        story = self._section("图件导读", "Figure Guide")
        story.append(_p(
            escape("本报告共 15 张图，覆盖整体结合拓扑、逐残基作用、几何与能量无关的描述性统计、"
                   "埋藏面积、残基组成、热点、网络拓扑、类型–类别对应、氢键联合几何、距离统计"
                   "与逐原子埋藏。建议先阅读图 1、图 2 和图 15，再按需查阅中间的专业图件。"
                   "每张图的图注与解读给出中英文说明和关键数值。"),
            self.styles["body_cn"],
        ))
        story.append(_p(
            escape("This report contains 15 figures covering overall binding topology, per-residue "
                   "engagement, geometry, buried surface area, residue composition, hotspots, network "
                   "topology, type–class correspondence, joint hydrogen-bond geometry, distance "
                   "statistics and per-atom burial. Figures 1, 2 and 15 provide a rapid overview; the "
                   "remaining figures provide specialized views. Bilingual captions and "
                   "interpretations accompany every figure."),
            self.styles["body_en"],
        ))
        head = [_p("图号<br/>Fig.", self.styles["table_head"]),
                _p("内容 / Content", self.styles["table_head"])]
        rows = [head]
        for spec in specs:
            short_cn, short_en = guide.get(spec["key"], ("", ""))
            content = _p(f"<b>{escape(short_cn)}</b><br/>"
                         f"<font size=7.4 color='#5B6570'>{escape(short_en)}</font>",
                         self.styles["table_cell"])
            rows.append([_p(str(spec["num"]), self.styles["table_cell_c"]), content])
        table = LongTable(rows, colWidths=[16 * mm, 174 * mm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
        ]))
        story.append(table)
        return story

    def _figure_story(self, spec: dict) -> list:
        path = self.fig_paths[spec["key"]]
        if spec.get("landscape"):
            image = _fit_image(path, 258 * mm, 124 * mm)
        else:
            image = _fit_image(path, 186 * mm, 188 * mm)
        image.hAlign = "CENTER"
        extra_cn, extra_en = self._figure_extra(spec["key"])
        cn = f'<b>图 {spec["num"]}.</b> {escape(spec["cn"])} {escape(extra_cn)}'
        en = f'<b>Figure {spec["num"]}.</b> {escape(spec["en"])} {escape(extra_en)}'
        icn = f'<b>解读：</b>{escape(spec["icn"])}'
        ien = f'<b>Interpretation:</b> {escape(spec["ien"])}'
        block = [Spacer(1, 2 * mm), image,
                 HRFlowable(width="100%", thickness=0.6, color=BORDER,
                            spaceBefore=4, spaceAfter=4)]
        if spec.get("landscape"):
            left = [_p(cn, self.styles["caption_cn"]),
                    _p(icn, self.styles["interp_cn"])]
            right = [_p(en, self.styles["caption_en"]),
                     _p(ien, self.styles["interp_en"])]
            cap = Table([[left, right]], colWidths=[129 * mm, 129 * mm])
            cap.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 5 * mm),
                ("LEFTPADDING", (1, 0), (1, 0), 5 * mm),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
            block.append(cap)
        else:
            block += [_p(cn, self.styles["caption_cn"]),
                      _p(en, self.styles["caption_en"]),
                      _p(icn, self.styles["interp_cn"]),
                      _p(ien, self.styles["interp_en"])]
        return block

    def _table(self, headers: list[str], rows: list[list[str]],
               widths: list[float], numeric_cols: set[int] | None = None) -> LongTable:
        numeric_cols = numeric_cols or set()
        head = []
        for h in headers:
            head.append(_p(escape(h), self.styles["table_head"]))
        body = []
        for row in rows:
            line = []
            for idx, value in enumerate(row):
                style = (self.styles["table_cell_c"] if idx in numeric_cols
                         else self.styles["table_cell"])
                line.append(_cell(value, style))
            body.append(line)
        table = LongTable([head] + body, colWidths=widths, repeatRows=1)
        commands = [
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]
        for row_idx in range(1, len(body) + 1):
            if row_idx % 2 == 1:
                commands.append(("BACKGROUND", (0, row_idx), (-1, row_idx), SOFT))
        table.setStyle(TableStyle(commands))
        return table

    def _tables_story(self) -> list:
        story = self._section("关键相互作用", "Key Interactions")
        story.append(_p(
            escape("下表列出按类型与距离排序的前 20 条相互作用记录；完整数据可通过 LigPlot-CN "
                   "导出 CSV 获取。原子对列中“—”表示该作用按残基对合并统计。"),
            self.styles["body_cn"],
        ))
        story.append(_p(
            escape("The table lists the top 20 records sorted by interaction type and distance. "
                   "The complete dataset can be exported as CSV from LigPlot-CN. An em dash in "
                   "the atom-pair column indicates residue-level aggregation."),
            self.styles["body_en"],
        ))
        if self.is_pl:
            headers = ["类型", "性质", "残基", "原子对", "距离 (Å)", "角度 (°)", "接触数"]
            widths = [18 * mm, 30 * mm, 32 * mm, 46 * mm, 16 * mm, 16 * mm, 16 * mm]
            rows = []
            for row in self.result.table_rows[:20]:
                typ, nature, residue, atom_a, atom_b, dist, angle, contacts = row
                atom_pair = "—" if atom_a == "—" and atom_b == "—" else f"{atom_a} – {atom_b}"
                rows.append([typ, nature, residue, atom_pair, dist, angle, contacts])
            story.append(self._table(headers, rows, widths, {4, 5, 6}))
        else:
            headers = ["类型", "性质", "残基 A", "残基 B", "原子对", "距离 (Å)", "角度 (°)", "接触"]
            widths = [16 * mm, 27 * mm, 27 * mm, 27 * mm, 37 * mm, 14 * mm, 13 * mm, 13 * mm]
            rows = []
            for row in self.result.table_rows[:20]:
                typ, nature, res_a, atom_a, res_b, atom_b, dist, angle, contacts = row
                atom_pair = "—" if atom_a == "—" and atom_b == "—" else f"{atom_a} – {atom_b}"
                rows.append([typ, nature, res_a, res_b, atom_pair, dist, angle, contacts])
            story.append(self._table(headers, rows, widths, {5, 6, 7}))

        story.append(PageBreak())
        story += self._section("残基定量汇总", "Residue-Level Metrics")
        story.append(_p(
            escape("每个残基的五类相互作用记录数、合计与最短接触距离；按合计记录数降序排列。"),
            self.styles["body_cn"],
        ))
        story.append(_p(
            escape("Interaction records by class, total records and minimum contact distance for "
                   "each residue, sorted by total records."),
            self.styles["body_en"],
        ))
        res_headers = ["残基", "氢键", "盐桥", "π–π", "配位", "疏水", "合计", "最短距离 (Å)"]
        res_widths = [42 * mm, 18 * mm, 18 * mm, 16 * mm, 16 * mm, 16 * mm, 16 * mm, 32 * mm]
        res_rows = [[str(v) for v in row] for row in self.result.residue_rows[:25]]
        story.append(self._table(res_headers, res_rows, res_widths, {1, 2, 3, 4, 5, 6, 7}))

        story.append(PageBreak())
        story += self._section("结合参数", "Binding Descriptors")
        descriptor_rows = self._descriptor_rows()
        story.append(self._table(["参数 / Parameter", "数值 / Value"],
                                 descriptor_rows,
                                 [110 * mm, 64 * mm], {1}))
        story += self._section("参数解释", "Descriptor Notes")
        for cn, en in self._descriptor_notes():
            story.append(_p(escape(cn), self.styles["body_cn"]))
            story.append(_p(escape(en), self.styles["body_en"]))
        return story

    def _descriptor_rows(self) -> list[list[str]]:
        a = self.assessment
        if self.is_pl:
            d = a.descriptors
            return [
                ["观测分子式 / Observed formula", d.formula or "—"],
                ["观测质量 / Observed mass (Da)", f"{d.mass:.2f}"],
                ["重原子数 / Heavy atoms", str(d.heavy_atoms)],
                ["环数（芳香环）/ Rings (aromatic)", f"{d.rings} ({d.aromatic_rings})"],
                ["可旋转键（近似）/ Rotatable bonds (approx.)", str(d.rotatable_bonds)],
                ["氢键供体 / 受体", f"{d.donors} / {d.acceptors}"],
                ["形式电荷（估计）/ Formal charge (est.)", f"{d.formal_charge:+d}"],
                ["疏水原子数 / Hydrophobic atoms", str(d.hydrophobic_atoms)],
                ["配体自由 SASA (Å²)", f"{d.sasa_free:.1f}"],
                ["配体结合态 SASA (Å²)", f"{d.sasa_bound:.1f}"],
                ["配体埋藏面积 (Å²)", f"{d.buried_area:.1f}"],
                ["配体埋藏比例 / Burial fraction", f"{100 * d.burial_fraction:.1f}%"],
                ["相互作用记录 / 重原子", f"{d.interactions_per_atom:.2f}"],
                ["结合埋藏面积 BSA 半和 (Å²)", f"{a.bsa:.1f}"],
                ["总埋藏面积 / Total buried (Å²)", f"{a.total_buried:.1f}"],
                ["每 100 Å² 互作记录", f"{a.interaction_density:.2f}"],
                ["每 100 Å² 氢键", f"{a.hbond_density:.2f}"],
            ]
        return [
            ["界面残基 链 A / B", f"{len(self.result.interface_a)} / {len(self.result.interface_b)}"],
            ["总埋藏面积 / Total buried (Å²)", f"{a.total_buried:.1f}"],
            ["界面面积 BSA 半和 (Å²)", f"{a.bsa:.1f}"],
            ["链 A 埋藏面积 (Å²)", f"{a.sasa.buried_a:.1f}"],
            ["链 B 埋藏面积 (Å²)", f"{a.sasa.buried_b:.1f}"],
            ["界面大小分类 / Size class", a.interface_size_class],
            ["每 100 Å² 互作记录", f"{a.interaction_density:.2f}"],
            ["每 100 Å² 氢键", f"{a.hbond_density:.2f}"],
            ["每 100 Å² 盐桥", f"{a.salt_density:.2f}"],
            ["疏水记录比例 / Hydrophobic fraction", f"{100 * a.hydrophobic_fraction:.1f}%"],
            ["疏水残基比例 / Hydrophobic residues",
             f"{a.composition_all.percentages['Hydrophobic']:.1f}%"],
            ["芳香残基比例 / Aromatic residues",
             f"{a.composition_all.percentages['Aromatic']:.1f}%"],
            ["极性残基比例 / Polar residues",
             f"{a.composition_all.percentages['Polar']:.1f}%"],
            ["带电残基比例 / Charged residues",
             f"{a.composition_all.percentages['Positive'] + a.composition_all.percentages['Negative']:.1f}%"],
        ]

    def _descriptor_notes(self) -> list[tuple[str, str]]:
        if self.is_pl:
            return [
                ("观测分子式与质量仅基于结构文件中实际建模的原子；若氢原子未被建模，"
                 "则质量与分子式为重原子观测值。可旋转键为基于键长与环系的近似计数；"
                 "供体/受体基于原子类型与局部化学环境推断。",
                 "Formula and mass are based only on atoms modelled in the structure; if hydrogens "
                 "are absent, the values are heavy-atom observations. Rotatable bonds are "
                 "approximated from bond lengths and ring membership; donor/acceptor counts are "
                 "inferred from atom types and local chemistry."),
                ("SASA 采用 Shrake–Rupley 数值积分（1.4 Å 探针、96 个球面点）。"
                 "不同软件、半径与采样密度会带来数值差异，建议用同一流程进行相对比较。",
                 "SASA was computed with Shrake–Rupley numerical integration (1.4 Å probe, 96 "
                 "sphere points). Different radii, probes and sampling densities will produce "
                 "slightly different values; relative comparisons should use the same workflow."),
            ]
        return [
            ("界面大小分类为经验性描述：总埋藏面积 <600 Å² 为小、600–1200 Å² 为中等、"
             ">1200 Å² 为大。该分类仅用于快速比较，不同研究中的定义可能不同。",
             "Interface-size classes are empirical: <600 Å² total buried as small, 600–1200 Å² as "
             "medium, and >1200 Å² as large. The classification is intended for rapid comparison; "
             "definitions vary across studies."),
            ("总埋藏面积是两侧埋藏面积之和；半和 BSA 为总埋藏的一半，是文献中常见的两侧界面面积定义。"
             "SASA 采用 Shrake–Rupley、1.4 Å 探针与 96 个球面点。",
             "Total buried surface is the sum of burial on both sides; half-sum BSA is half of that "
             "value and is a common definition of two-sided interface area. SASA used "
             "Shrake–Rupley with a 1.4 Å probe and 96 sphere points."),
        ]

    def _methods_story(self) -> list:
        story = self._section("方法与参数", "Methods and Parameters")
        p = self.result.params
        cn_method = (
            f"结构解析：仅读取第一个 MODEL，去除水分子；氢键采用供体–受体重原子距离 "
            f"≤ {p.hbond_dist:.1f} Å 与 D–H···A 角度 ≥ {p.hbond_angle:.0f}°（无显式氢时使用虚拟氢）；"
            f"疏水接触采用侧链 C/S 原子距离 ≤ {p.hydrophobic_dist:.1f} Å；"
            f"盐桥采用正负带电基团距离 ≤ {p.salt_dist:.1f} Å，并按残基对合并；"
            f"π–π 采用芳环质心距离 ≤ {p.pi_dist:.1f} Å，区分平行（≤30°）与 T 型（≥60°）；"
            f"金属配位采用配体金属与蛋白 N/O/S 距离 ≤ {p.metal_dist:.1f} Å。"
            f"蛋白–蛋白界面残基按任意重原子距离 ≤ {p.interface_dist:.1f} Å 判定。"
            f"SASA/BSA 采用 Shrake–Rupley 方法（探针 1.4 Å，96 个球面点）。"
        )
        en_method = (
            f"Structure parsing: only the first MODEL was read and waters were excluded. "
            f"Hydrogen bonds used a donor–acceptor distance ≤ {p.hbond_dist:.1f} Å and a "
            f"D–H···A angle ≥ {p.hbond_angle:.0f}° (virtual hydrogens when absent). Hydrophobic "
            f"contacts used side-chain C/S atoms ≤ {p.hydrophobic_dist:.1f} Å. Salt bridges used "
            f"oppositely charged groups ≤ {p.salt_dist:.1f} Å and were merged per residue pair. "
            f"π–π stacking used ring-centroid distances ≤ {p.pi_dist:.1f} Å, classified as parallel "
            f"(≤30°) or T-shaped (≥60°). Metal coordination used ligand metals and protein N/O/S "
            f"atoms ≤ {p.metal_dist:.1f} Å. Protein–protein interface residues were defined by any "
            f"heavy-atom distance ≤ {p.interface_dist:.1f} Å. SASA/BSA used Shrake–Rupley with a "
            f"1.4 Å probe and 96 sphere points."
        )
        story.append(_p(escape(cn_method), self.styles["body_cn"]))
        story.append(_p(escape(en_method), self.styles["body_en"]))

        story += self._section("评价与局限", "Assessment and Limitations")
        limits_cn = (
            "本报告为基于单一静态结构的描述性分析。相互作用判据为常用几何阈值，"
            "不同阈值会改变记录数量；氢键方向依赖虚拟氢或显式氢，可能带来假阳性/假阴性。"
            "观察到的作用数量与埋藏面积不等同于结合自由能或亲和力；"
            "金属配位、π–π 与盐桥的判定为经验规则。"
            "如需定量比较不同复合物，建议保持相同参数与同一软件版本，并结合实验验证。"
        )
        limits_en = (
            "This report is a descriptive analysis of a single static structure. Interaction "
            "criteria are geometric thresholds commonly used in the field; different cutoffs will "
            "change the number of records. Hydrogen-bond direction depends on explicit or virtual "
            "hydrogens and may introduce false positives or negatives. Counts and buried areas are "
            "not equivalent to binding free energy or affinity. Metal coordination, π–π and salt "
            "bridge assignments are heuristic. For quantitative comparisons across complexes, keep "
            "identical parameters and software version and validate with experimental data."
        )
        story.append(_p(escape(limits_cn), self.styles["body_cn"]))
        story.append(_p(escape(limits_en), self.styles["body_en"]))

        story += self._section("参考", "References")
        refs = [
            "Wallace AC, Laskowski RA, Thornton JM. LIGPLOT: a program to generate schematic "
            "diagrams of protein–ligand interactions. Protein Engineering. 1995;8:127–134.",
            "Laskowski RA, Swindells MB. LigPlot+: multiple ligand–protein interaction diagrams "
            "for drug discovery. Journal of Chemical Information and Modeling. 2011;51:2778–2786.",
            "Salentin S, Schreiber S, Haupt VJ, Adasme MF, Schroeder M. PLIP: fully automated "
            "protein–ligand interaction profiler. Nucleic Acids Research. 2015;43:W443–W447.",
            "Shrake A, Rupley JA. Environment and exposure to solvent of protein atoms. "
            "Lysozyme and insulin. Journal of Molecular Biology. 1973;79:351–371.",
        ]
        for ref in refs:
            story.append(_p(escape(ref), self.styles["ref"]))
        return story

    # ------------------------------------------------------------------
    # 页面
    # ------------------------------------------------------------------
    def _on_cover(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        band_h = 100 * mm
        canvas.setFillColor(PRIMARY)
        canvas.rect(0, PAGE_H - band_h, PAGE_W, band_h, fill=1, stroke=0)
        canvas.setStrokeColor(GOLD)
        canvas.setLineWidth(1.1)
        canvas.line(MARGIN_L, PAGE_H - band_h, PAGE_W - MARGIN_R, PAGE_H - band_h)

        canvas.setFillColor(colors.HexColor("#A9C9B7"))
        canvas.setFont(FONT_ENB, 9.5)
        canvas.drawString(MARGIN_L, PAGE_H - 17 * mm, "LIGPLOT-CN")
        canvas.setFont(FONT_CN, 7.2)
        canvas.drawString(MARGIN_L, PAGE_H - 22 * mm,
                          "STRUCTURAL INTERACTION REPORT  ·  结构互作分析报告")

        canvas.setFillColor(colors.white)
        canvas.setFont(FONT_EN, 8.5)
        canvas.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 17 * mm,
                               f"PDB {self.result.structure.pdb_id}")
        canvas.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 22 * mm, self._entity())

        title, title_cn, _ = self._title()
        lines = title.split(" ")
        canvas.setFillColor(colors.white)
        canvas.setFont(FONT_SERIFB, 26)
        canvas.drawString(MARGIN_L, PAGE_H - 54 * mm, lines[0])
        if len(lines) > 1:
            canvas.drawString(MARGIN_L, PAGE_H - 66 * mm, " ".join(lines[1:]))
        canvas.setFillColor(colors.HexColor("#DDEBDD"))
        canvas.setFont(FONT_CN, 13)
        canvas.drawString(MARGIN_L, PAGE_H - 81 * mm, title_cn)
        canvas.setFont(FONT_EN, 8.2)
        canvas.setFillColor(colors.HexColor("#A9C9B7"))
        canvas.drawString(MARGIN_L, PAGE_H - 88 * mm,
                          f"{self._entity()}   ·   {len(self.result.interactions)} interaction records")

        hero = self.fig_paths.get("main")
        if hero:
            iw, ih = PILImage.open(hero).size
            max_w, max_h = 160 * mm, 100 * mm
            ratio = min(max_w / iw, max_h / ih)
            w, h = iw * ratio, ih * ratio
            x = (PAGE_W - w) / 2
            y = 92 * mm
            canvas.setFillColor(colors.HexColor("#FBFDFB"))
            canvas.setStrokeColor(BORDER)
            canvas.setLineWidth(0.8)
            canvas.roundRect(x - 3 * mm, y - 3 * mm, w + 6 * mm, h + 6 * mm,
                             3 * mm, stroke=1, fill=1)
            canvas.drawImage(hero, x, y, width=w, height=h, mask="auto")

        # 元数据
        a = self.assessment
        meta = [
            ("结构 / Structure", f"PDB {self.result.structure.pdb_id}"),
            ("模式 / Mode", "蛋白–小分子 / Protein–ligand"
             if self.is_pl else "蛋白–蛋白 / Protein–protein"),
            ("对象 / Entity", self._entity()),
            ("相互作用 / Interactions", f"{len(self.result.interactions)} records"),
            ("界面面积 / BSA (Å²)", f"{a.bsa:.1f}  (half-sum)"),
            ("报告日期 / Date", self.date),
        ]
        canvas.setFont(FONT_CN, 6.6)
        for idx, (label, value) in enumerate(meta):
            col = idx % 2
            row = idx // 2
            x = MARGIN_L + col * 88 * mm
            y = 75 * mm - row * 10 * mm
            canvas.setFillColor(MUTED)
            canvas.drawString(x, y, label)
            canvas.setFillColor(TEXT)
            canvas.setFont(FONT_CN, 8.0)
            canvas.drawString(x, y - 4.6 * mm, value)
            canvas.setFont(FONT_CN, 6.6)

        canvas.setFillColor(colors.HexColor("#3F5147"))
        canvas.setFont(FONT_CN, 7.4)
        canvas.drawString(MARGIN_L, 37 * mm,
                          "本报告包含 15 张出版级图件，全部图注提供中英文解释。")
        canvas.setFillColor(MUTED)
        canvas.setFont(FONT_EN, 6.6)
        canvas.drawString(MARGIN_L, 32 * mm,
                          "Fifteen publication-quality figures with bilingual Chinese/English captions.")

        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.6)
        canvas.line(MARGIN_L, 17 * mm, PAGE_W - MARGIN_R, 17 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont(FONT_EN, 7.0)
        canvas.drawString(MARGIN_L, 11 * mm,
                          f"Generated by LigPlot-CN v{__version__}  ·  {self.date}")
        canvas.drawRightString(PAGE_W - MARGIN_R, 11 * mm,
                               "Structural Interaction Report")
        canvas.restoreState()

    def _on_content(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.6)
        canvas.line(MARGIN_L, PAGE_H - 16 * mm, PAGE_W - MARGIN_R, PAGE_H - 16 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont(FONT_EN, 6.8)
        canvas.drawString(MARGIN_L, PAGE_H - 12 * mm,
                          "LigPlot-CN  ·  Structural Interaction Report")
        canvas.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 12 * mm,
                               f"PDB {self.result.structure.pdb_id}  ·  {self._entity()}")

        canvas.setStrokeColor(BORDER)
        canvas.line(MARGIN_L, 15 * mm, PAGE_W - MARGIN_R, 15 * mm)
        canvas.setFont(FONT_EN, 6.8)
        canvas.drawString(MARGIN_L, 10 * mm,
                          f"Generated {self.date}  ·  LigPlot-CN v{__version__}")
        canvas.drawRightString(PAGE_W - MARGIN_R, 10 * mm, str(canvas.getPageNumber()))
        canvas.restoreState()

    def _on_landscape(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.6)
        canvas.line(MARGIN_L, PAGE_H_L - 16 * mm, PAGE_W_L - MARGIN_R, PAGE_H_L - 16 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont(FONT_EN, 6.8)
        canvas.drawString(MARGIN_L, PAGE_H_L - 12 * mm,
                          "LigPlot-CN  ·  Structural Interaction Report")
        canvas.drawRightString(PAGE_W_L - MARGIN_R, PAGE_H_L - 12 * mm,
                               f"PDB {self.result.structure.pdb_id}  ·  {self._entity()}")
        canvas.setStrokeColor(BORDER)
        canvas.line(MARGIN_L, 14 * mm, PAGE_W_L - MARGIN_R, 14 * mm)
        canvas.setFont(FONT_EN, 6.8)
        canvas.drawString(MARGIN_L, 9 * mm,
                          f"Generated {self.date}  ·  LigPlot-CN v{__version__}")
        canvas.drawRightString(PAGE_W_L - MARGIN_R, 9 * mm, str(canvas.getPageNumber()))
        canvas.restoreState()

    def _build_pdf(self):
        doc = BaseDocTemplate(
            self.output_path, pagesize=A4,
            leftMargin=MARGIN_L, rightMargin=MARGIN_R,
            topMargin=MARGIN_T, bottomMargin=MARGIN_B,
            title=f"LigPlot-CN Interaction Report — {self.result.structure.pdb_id}",
            author="LigPlot-CN", subject="Protein interaction analysis report",
            creator=f"LigPlot-CN v{__version__}",
        )
        cover_frame = Frame(0, 0, PAGE_W, PAGE_H, id="cover",
                            leftPadding=0, rightPadding=0,
                            topPadding=0, bottomPadding=0)
        content_frame = Frame(MARGIN_L, MARGIN_B,
                              PAGE_W - MARGIN_L - MARGIN_R,
                              PAGE_H - MARGIN_T - MARGIN_B,
                              id="content", leftPadding=0, rightPadding=0,
                              topPadding=0, bottomPadding=0)
        figure_frame = Frame(12 * mm, MARGIN_B,
                             PAGE_W - 24 * mm,
                             PAGE_H - MARGIN_T - MARGIN_B,
                             id="figure", leftPadding=0, rightPadding=0,
                             topPadding=0, bottomPadding=0)
        landscape_frame = Frame(MARGIN_L, 16 * mm,
                                PAGE_W_L - MARGIN_L - MARGIN_R,
                                PAGE_H_L - 36 * mm,
                                id="landscape", leftPadding=0, rightPadding=0,
                                topPadding=0, bottomPadding=0)
        doc.addPageTemplates([
            PageTemplate(id="cover", frames=[cover_frame], onPage=self._on_cover),
            PageTemplate(id="content", frames=[content_frame], onPage=self._on_content),
            PageTemplate(id="figure", frames=[figure_frame], onPage=self._on_content),
            PageTemplate(id="landscape", frames=[landscape_frame],
                         onPage=self._on_landscape, pagesize=landscape(A4)),
        ])

        story = [Spacer(1, 1), NextPageTemplate("content"), PageBreak()]
        story += self._summary_story()

        story.append(PageBreak())
        story += self._figure_guide_story()

        specs = self._figure_specs()
        for spec in specs:
            if spec.get("landscape"):
                template = "landscape"
            else:
                template = "figure"
            story.append(NextPageTemplate(template))
            story.append(PageBreak())
            story += self._figure_story(spec)

        story.append(NextPageTemplate("content"))
        story.append(PageBreak())
        story += self._tables_story()
        story.append(PageBreak())
        story += self._methods_story()

        self.progress("写入 PDF…")
        doc.build(story)

    def generate(self) -> str:
        self.progress("计算 SASA / BSA 与描述符…")
        if self.is_pl:
            self.assessment = analyze_pl_binding(self.result, progress=self.progress)
        else:
            self.assessment = analyze_pp_binding(self.result, progress=self.progress)
        self._make_figures()
        self._build_pdf()
        self.progress("报告生成完成")
        return self.output_path


def generate_report(result: PLResult | PPResult, output_path: str,
                    progress=None) -> str:
    """生成中英双语 PDF 报告并返回路径。"""
    report = InteractionReport(result, output_path, progress=progress)
    try:
        return report.generate()
    finally:
        report.tmpdir.cleanup()
