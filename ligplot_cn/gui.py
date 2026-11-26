"""LigPlot-CN 图形界面（中文 UI，英文图注）。

界面：深绿顶栏 + 卡片式侧栏 + 主图/定量图切换 + KPI 指标条 + 数据面板。
"""

from __future__ import annotations

import csv
import os
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import __version__
from .analysis import KIND_NAMES_CN, PLResult, PPResult, analyze_pl, analyze_pp
from .interactions import InteractionParams
from .pdbio import Structure, parse_structure
from .quant import build_quant_figure
from .render import build_pl_figure, build_pp_figure

UI_FONT = "Microsoft YaHei UI"

INK = "#17251E"
TEXT = "#26352D"
MUTED = "#6C7B72"
BG = "#EDF2EE"
CARD = "#FFFFFF"
SOFT = "#F4F8F5"
BORDER = "#DCE6DF"
PRIMARY = "#2C6E52"
PRIMARY_DARK = "#174534"
PRIMARY_HOVER = "#3A8564"
MINT = "#A9C9B7"
GOLD = "#C8A24A"

KIND_COLOR = {
    "hbond": "#2E7D32",
    "salt": "#7B1FA2",
    "pi": "#00796B",
    "metal": "#6D4C41",
    "hydrophobic": "#AE3B2C",
}

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES = {
    "1A4W": (BASE_DIR / "samples" / "1A4W.pdb", "1A4W · 蛋白–小分子（PDB）"),
    "1A4W_CIF": (BASE_DIR / "samples" / "1A4W.cif", "1A4W · 蛋白–小分子（mmCIF）"),
    "1BRS": (BASE_DIR / "samples" / "1BRS.pdb", "1BRS · 蛋白–蛋白（PDB）"),
}

TAG_BY_KIND_CN = {
    "氢键": "hbond",
    "盐桥": "salt",
    "π–π 堆积": "pi",
    "配位键": "metal",
    "疏水接触": "hydrophobic",
}

APP_ICON_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAACJElEQVR4nO2bvU7DMBSFTyueAcRb"
    "dGABZlY6UiHxCh2rPkLFyCtUQjDC2hlYOvAWqH0JmByZ2I5vEtvXf99U5ce95+T4JmkaoFI2kyE7"
    "nd1c/LouxBWH3b6XJvLGMYs2QTFjShkoRfEAre5Oh1IVrsOUBmMCchIPmPVoDchNvECni9QDckYx"
    "INejL2jrm3atzBVZZ50C3AVw0xhQSvwFQm/xCTgZstPn82vz+er+zlkxHPQyQBbeXpaqEYMSEBtj"
    "EtncINiaoO7otwmdgq6aKLUcdvtJ8U0wWQNsiaQkFuhhgC1S798fWK+W1OGiwXkC1qtlUkaQm6B"
    "Mu+uaBG8en8bUpkX+rtvZtXE7ahMcZIAJnRGuTNCNHZ0BgL5QYJwRtjGHXgd4MQCgmfD2cKqsn2"
    "+PvccZgzcDBKYpoRMvECaE6CveDQBUIZc/L9Z9vs4XyjIfDTXIlaBcOEW8bjsf4gXeEyDoir2Jd"
    "k9wTb0XQML3Aq6oBnAXwE0wA/o2NN8NUBA0AVRRocQDDFPAJi6keICpB5hEhhYPMP4qPN8e/10m"
    "+7za66KeBbgL4KYawF0AN9UA7gK4YX04Kv+quwHPaZDFgJgesxc/BYIb4OqhpitqArgL4KYxoO+b"
    "FkOxNblQTVDoLT4BLKdBcZRj+LudEvsS/jEqT3dlCoTqBVy09RXfA7QG5JoCnS5jAnIzwaSHJDL"
    "lxmg7kKQekGoaKHUX/+5w8fwB1ynehdydTCUAAAAASUVORK5CYII="
)


def _enable_dpi_awareness() -> None:
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def _apply_dark_titlebar(root: tk.Tk) -> None:
    try:
        import ctypes
        root.update_idletasks()
        hwnd = int(root.wm_frame(), 16)
        dwm = ctypes.windll.dwmapi
        for attr, color in ((35, 0x00344517), (36, 0x00FFFFFF)):
            value = ctypes.c_int(color)
            dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(value),
                                      ctypes.sizeof(value))
    except Exception:
        pass


class ChineseToolbar(NavigationToolbar2Tk):
    toolitems = [
        ("主页", "重置为初始视图", "home", "home"),
        ("后退", "返回上一个视图", "back", "back"),
        ("前进", "前进到下一个视图", "forward", "forward"),
        ("平移", "按住左键拖动平移", "move", "pan"),
        ("缩放", "框选区域放大", "zoom_to_rect", "zoom"),
        ("保存", "保存当前视图", "filesave", "save_figure"),
    ]

    def __init__(self, canvas, window, **kwargs):
        super().__init__(canvas, window, pack_toolbar=False, **kwargs)
        try:
            self.config(background=CARD)
        except Exception:
            pass

    def set_message(self, s):
        # 不在工具栏显示鼠标坐标，保持界面简洁
        return


class LigPlotApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        try:
            self._icon = tk.PhotoImage(data=APP_ICON_B64)
            self.root.iconphoto(True, self._icon)
        except Exception:
            self._icon = None
        self.structure: Structure | None = None
        self.result: PLResult | PPResult | None = None
        self.last_mode: str | None = None
        self.last_show_distances = True
        self.view_mode = "diagram"
        self.ligand_map: dict[str, object] = {}
        self.chain_display_map: dict[str, str] = {}
        self.fig_canvas: FigureCanvasTkAgg | None = None
        self.toolbar: ChineseToolbar | None = None
        self.kpi_labels: dict[str, tk.Label] = {}

        self._setup_theme()
        self._build_vars()
        self._build_menu()
        self._build_layout()
        self._set_status("就绪：打开 PDB / mmCIF 结构，或从“示例”载入演示数据。")

    # ------------------------------------------------------------------
    # 主题与基础组件
    # ------------------------------------------------------------------
    def _setup_theme(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", font=(UI_FONT, 9), background=BG, foreground=TEXT)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("Soft.TFrame", background=SOFT)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Card.TLabel", background=CARD, foreground=TEXT)
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED,
                        font=(UI_FONT, 8))
        style.configure("Section.TLabel", background=CARD, foreground=INK,
                        font=(UI_FONT, 10, "bold"))
        style.configure("Sub.TLabel", background=CARD, foreground="#3E5749",
                        font=(UI_FONT, 8, "bold"))
        style.configure("FileTitle.TLabel", background=CARD, foreground=INK,
                        font=(UI_FONT, 10, "bold"), wraplength=318, justify="left")
        style.configure("FileMeta.TLabel", background=CARD, foreground=MUTED,
                        font=(UI_FONT, 8), wraplength=318, justify="left")
        style.configure("Mode.TNotebook", background=CARD, borderwidth=0,
                        tabmargins=(0, 0, 0, 0))
        style.configure("Mode.TNotebook.Tab", padding=(11, 6), font=(UI_FONT, 9),
                        background="#EDF2EE", foreground="#5A6B61")
        style.map("Mode.TNotebook.Tab",
                  background=[("selected", CARD)],
                  foreground=[("selected", PRIMARY)])
        style.configure("Result.TNotebook", background=CARD, borderwidth=0,
                        tabmargins=(0, 0, 0, 0))
        style.configure("Result.TNotebook.Tab", padding=(12, 5), font=(UI_FONT, 8),
                        background=SOFT, foreground="#5A6B61")
        style.map("Result.TNotebook.Tab",
                  background=[("selected", CARD)],
                  foreground=[("selected", PRIMARY)])
        style.configure("Primary.TButton", background=PRIMARY, foreground="white",
                        borderwidth=0, padding=(12, 8),
                        font=(UI_FONT, 10, "bold"))
        style.map("Primary.TButton",
                  background=[("active", PRIMARY_HOVER), ("disabled", "#B7C7BE")],
                  foreground=[("disabled", "#F0F4F1")])
        style.configure("Secondary.TButton", background=SOFT, foreground=TEXT,
                        borderwidth=0, padding=(10, 6), font=(UI_FONT, 9))
        style.map("Secondary.TButton",
                  background=[("active", "#E6EFE9"), ("disabled", "#F5F7F5")],
                  foreground=[("disabled", "#B0BBB4")])
        style.configure("Ghost.TButton", background=CARD, foreground=PRIMARY,
                        borderwidth=0, padding=(4, 4), font=(UI_FONT, 8))
        style.map("Ghost.TButton",
                  background=[("active", SOFT), ("disabled", CARD)],
                  foreground=[("disabled", "#B0BBB4")])
        style.configure("Small.TButton", background=SOFT, foreground=TEXT,
                        borderwidth=0, padding=(7, 4), font=(UI_FONT, 8))
        style.map("Small.TButton", background=[("active", "#E6EFE9")])
        style.configure("Report.TButton", background="#EFE4C8", foreground="#3A2E12",
                        borderwidth=0, padding=(10, 7),
                        font=(UI_FONT, 9, "bold"))
        style.map("Report.TButton",
                  background=[("active", "#E6D7B2"), ("disabled", "#F6F2E7")],
                  foreground=[("disabled", "#BDB49E")])
        style.configure("View.TButton", background=SOFT, foreground=MUTED,
                        borderwidth=0, padding=(10, 5), font=(UI_FONT, 8))
        style.map("View.TButton", background=[("active", "#E6EFE9")])
        style.configure("ViewActive.TButton", background=PRIMARY, foreground="white",
                        borderwidth=0, padding=(10, 5), font=(UI_FONT, 8, "bold"))
        style.map("ViewActive.TButton",
                  background=[("active", PRIMARY_HOVER), ("disabled", "#B7C7BE")],
                  foreground=[("disabled", "#EEF3F0")])
        style.configure("Data.Treeview", background=CARD, fieldbackground=CARD,
                        rowheight=25, font=(UI_FONT, 8), borderwidth=0)
        style.configure("Data.Treeview.Heading", background=SOFT,
                        foreground="#41554A", font=(UI_FONT, 8, "bold"),
                        padding=(3, 5), relief="flat")
        style.map("Data.Treeview",
                  background=[("selected", "#DCEDE2")],
                  foreground=[("selected", INK)])
        style.configure("TSpinbox", fieldbackground=CARD, arrowsize=12,
                        padding=(3, 3))
        style.configure("TCombobox", fieldbackground=CARD, padding=(3, 3))
        style.configure("Card.TCheckbutton", background=CARD, font=(UI_FONT, 8))
        style.map("Card.TCheckbutton", background=[("active", CARD)])
        style.configure("TCheckbutton", background=CARD, font=(UI_FONT, 8))

    def _build_vars(self):
        self.var_hbond_dist = tk.DoubleVar(value=3.5)
        self.var_hbond_angle = tk.DoubleVar(value=100.0)
        self.var_hydro_dist = tk.DoubleVar(value=4.0)
        self.var_interface_dist = tk.DoubleVar(value=4.0)
        self.var_pi = tk.BooleanVar(value=True)
        self.var_show_dist = tk.BooleanVar(value=True)
        self.var_mode = tk.IntVar(value=0)
        self.var_ligand_info = tk.StringVar(value="未选择配体")
        self.var_chain_a = tk.StringVar()
        self.var_chain_b = tk.StringVar()
        self.var_file_state = tk.StringVar(value="○ 未载入")

    def _build_menu(self):
        menu = tk.Menu(self.root, tearoff=False)
        sample_menu = tk.Menu(menu, tearoff=False)
        for key, (path, label) in SAMPLES.items():
            sample_menu.add_command(label=label,
                                    command=lambda k=key: self._load_sample(k))
        menu.add_cascade(label="打开示例结构", menu=sample_menu)
        menu.add_separator()
        menu.add_command(label="生成 PDF 报告…", accelerator="Ctrl+P",
                         command=self._export_report)
        menu.add_command(label="导出当前图…", accelerator="Ctrl+S",
                         command=self._export_figure)
        menu.add_command(label="导出数据表（CSV）…", command=self._export_csv)
        menu.add_separator()
        menu.add_command(label="视图：互作图", accelerator="Ctrl+1",
                         command=lambda: self._set_view("diagram"))
        menu.add_command(label="视图：定量图", accelerator="Ctrl+2",
                         command=lambda: self._set_view("quant"))
        menu.add_command(label="适应窗口", accelerator="Ctrl+0",
                         command=self._refit_preview)
        menu.add_separator()
        menu.add_command(label="使用说明", command=self._show_help)
        menu.add_command(label="关于 LigPlot-CN", command=self._show_about)
        menu.add_separator()
        menu.add_command(label="退出", command=self.root.destroy)
        self.more_menu = menu

        self.root.bind("<Control-o>", lambda e: self._open_dialog())
        self.root.bind("<Control-s>", lambda e: self._export_figure())
        self.root.bind("<Control-p>", lambda e: self._export_report())
        self.root.bind("<Control-1>", lambda e: self._set_view("diagram"))
        self.root.bind("<Control-2>", lambda e: self._set_view("quant"))
        self.root.bind("<Control-0>", lambda e: self._refit_preview())

    def _build_layout(self):
        self.root.title(f"LigPlot-CN {__version__}  ·  蛋白相互作用可视化与定量分析")
        self.root.geometry("1480x940")
        self.root.minsize(1180, 800)
        self.root.configure(background=BG)

        self._build_header()

        status = tk.Frame(self.root, bg="#E3EBE5", height=26)
        status.pack(fill="x", side="bottom")
        status.pack_propagate(False)
        self.status_var = tk.StringVar(value="")
        tk.Label(status, textvariable=self.status_var, bg="#E3EBE5", fg="#4E6055",
                 font=(UI_FONT, 8), anchor="w").pack(side="left", padx=12)
        tk.Label(status, text=f"LigPlot-CN {__version__}  ·  PDB / mmCIF  ·  600 dpi export",
                 bg="#E3EBE5", fg="#7A8A80", font=(UI_FONT, 8)).pack(
            side="right", padx=12)

        body = ttk.Frame(self.root, padding=(14, 12, 14, 0))
        body.pack(fill="both", expand=True)
        paned = ttk.PanedWindow(body, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, width=372)
        paned.add(left, weight=0)
        right = ttk.PanedWindow(paned, orient="vertical")
        self.right_paned = right
        paned.add(right, weight=1)

        self._build_sidebar(left)
        self._build_figure_card(right)
        self._build_results_card(right)
        self._update_view_buttons()
        self.root.after(500, self._set_initial_sash)
        self.root.after(300, self._show_placeholder)

    def _build_header(self):
        header = tk.Frame(self.root, bg=PRIMARY_DARK, height=66)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        logo = tk.Canvas(header, width=44, height=44, bg=PRIMARY_DARK,
                         highlightthickness=0)
        logo.pack(side="left", padx=(16, 6), pady=11)
        cx, cy = 22, 22
        for nx, ny in ((10, 13), (34, 15), (22, 36)):
            logo.create_line(cx, cy, nx, ny, fill="#8FBBA2", width=2)
        logo.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill=GOLD, outline="")
        for nx, ny in ((10, 13), (34, 15), (22, 36)):
            logo.create_oval(nx - 3, ny - 3, nx + 3, ny + 3,
                             fill="#DDEBDD", outline="#8FBBA2")

        title_box = tk.Frame(header, bg=PRIMARY_DARK)
        title_box.pack(side="left", pady=10)
        tk.Label(title_box, text="LigPlot-CN", bg=PRIMARY_DARK, fg="white",
                 font=(UI_FONT, 15, "bold")).pack(anchor="w")
        tk.Label(title_box, text="蛋白相互作用可视化与定量分析 · Protein interaction atlas",
                 bg=PRIMARY_DARK, fg=MINT, font=(UI_FONT, 8)).pack(anchor="w")

        right = tk.Frame(header, bg=PRIMARY_DARK)
        right.pack(side="right", padx=(0, 16))
        more = tk.Menubutton(right, text="更多 ▾", menu=self.more_menu,
                             bg=PRIMARY_DARK, fg=MINT, activebackground="#245641",
                             activeforeground="white", relief="flat", bd=0,
                             padx=10, pady=7, font=(UI_FONT, 9), cursor="hand2",
                             highlightthickness=0, indicatoron=False)
        more.pack(side="right")
        self._header_button(right, "PDF 报告", self._export_report, style="gold").pack(
            side="right", padx=(6, 0))
        self._header_button(right, "使用说明", self._show_help, style="ghost").pack(
            side="right", padx=(6, 0))
        self._header_button(right, "导出当前图", self._export_figure, style="light").pack(
            side="right", padx=(6, 0))
        self._header_button(right, "打开结构", self._open_dialog, style="outline").pack(
            side="right")

    def _header_button(self, parent, text, command, style="outline"):
        if style == "light":
            bg, fg, hover, border = "#DCEADF", PRIMARY_DARK, "#CBE0D1", ""
        elif style == "gold":
            bg, fg, hover, border = "#E7D9B5", "#3A2E12", "#DECBA0", ""
        elif style == "ghost":
            bg, fg, hover, border = PRIMARY_DARK, MINT, "#245641", ""
        else:
            bg, fg, hover, border = "#1E5741", "white", "#286A50", "#4E7C65"
        btn = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                        activebackground=hover, activeforeground=fg,
                        relief="flat", bd=0, padx=14, pady=7,
                        font=(UI_FONT, 9), cursor="hand2",
                        highlightthickness=1 if border else 0,
                        highlightbackground=border or PRIMARY_DARK)
        btn.bind("<Enter>", lambda e: btn.configure(bg=hover))
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg))
        return btn

    # ------------------------------------------------------------------
    # 侧栏
    # ------------------------------------------------------------------
    def _make_card(self, parent, pad=12):
        outer = tk.Frame(parent, bg=CARD, highlightbackground=BORDER,
                         highlightthickness=1)
        inner = ttk.Frame(outer, style="Card.TFrame", padding=pad)
        inner.pack(fill="both", expand=True)
        return outer, inner

    def _section_header(self, parent, text: str, right_text: str | None = None):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 8))
        bar = tk.Frame(row, bg=PRIMARY, width=3, height=14)
        bar.pack(side="left", padx=(0, 7))
        ttk.Label(row, text=text, style="Section.TLabel").pack(side="left")
        if right_text:
            ttk.Label(row, text=right_text, style="Muted.TLabel").pack(side="right")

    def _build_sidebar(self, parent):
        file_card, file_inner = self._make_card(parent, pad=13)
        file_card.pack(fill="x", pady=(0, 10))
        head = ttk.Frame(file_inner, style="Card.TFrame")
        head.pack(fill="x", pady=(0, 7))
        bar = tk.Frame(head, bg=PRIMARY, width=3, height=14)
        bar.pack(side="left", padx=(0, 7))
        ttk.Label(head, text="结构文件", style="Section.TLabel").pack(side="left")
        self.lbl_file_state = tk.Label(head, textvariable=self.var_file_state,
                                       bg=CARD, fg="#95A39A", font=(UI_FONT, 8))
        self.lbl_file_state.pack(side="right")

        self.lbl_title = ttk.Label(file_inner, text="尚未载入结构",
                                   style="FileTitle.TLabel")
        self.lbl_title.pack(fill="x", pady=(0, 3))
        self.lbl_meta = ttk.Label(file_inner,
                                  text="支持 PDB / mmCIF，可 gzip 压缩",
                                  style="FileMeta.TLabel")
        self.lbl_meta.pack(fill="x", pady=(0, 9))

        row = ttk.Frame(file_inner, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Button(row, text="打开结构文件…", style="Primary.TButton",
                   command=self._open_dialog).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="示例 ▾", style="Secondary.TButton",
                   command=self._sample_menu_popup).pack(side="left", padx=(6, 0))

        action_card, action_inner = self._make_card(parent, pad=13)
        action_card.pack(side="bottom", fill="x", pady=(10, 0))
        self.btn_analyze = ttk.Button(action_inner, text="分析并绘图",
                                      style="Primary.TButton", command=self._run_analysis)
        self.btn_analyze.pack(fill="x")
        row2 = ttk.Frame(action_inner, style="Card.TFrame")
        row2.pack(fill="x", pady=(6, 0))
        self.btn_export_png = ttk.Button(row2, text="导出当前图", style="Secondary.TButton",
                                         command=self._export_figure, state="disabled")
        self.btn_export_png.pack(side="left", fill="x", expand=True)
        self.btn_export_csv = ttk.Button(row2, text="导出数据表", style="Secondary.TButton",
                                         command=self._export_csv, state="disabled")
        self.btn_export_csv.pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Label(action_inner, text="界面中文 · 图注英文 · 600 dpi / PDF / SVG",
                  style="Muted.TLabel").pack(fill="x", pady=(7, 0))
        mode_card, mode_inner = self._make_card(parent, pad=10)
        mode_card.pack(fill="both", expand=True)
        ttk.Label(mode_inner, text="分析模式", style="Section.TLabel").pack(
            anchor="w", pady=(0, 6))
        self.nb_mode = ttk.Notebook(mode_inner, style="Mode.TNotebook")
        self.nb_mode.pack(fill="both", expand=True)
        self.nb_mode.bind("<<NotebookTabChanged>>", self._on_mode_changed)
        self.tab_pl = ttk.Frame(self.nb_mode, style="Card.TFrame", padding=(8, 8, 8, 4))
        self.tab_pp = ttk.Frame(self.nb_mode, style="Card.TFrame", padding=(8, 8, 8, 4))
        self.nb_mode.add(self.tab_pl, text="  蛋白–小分子  ")
        self.nb_mode.add(self.tab_pp, text="  蛋白–蛋白  ")
        self._build_pl_tab(self.tab_pl)
        self._build_pp_tab(self.tab_pp)

    def _build_pl_tab(self, parent):
        ttk.Label(parent, text="配体选择", style="Sub.TLabel").pack(anchor="w")
        wrap = tk.Frame(parent, bg=CARD, highlightbackground=BORDER,
                        highlightthickness=1)
        wrap.pack(fill="x", pady=(4, 3))
        cols = ("name", "chain", "seq", "atoms")
        self.lig_tree = ttk.Treeview(wrap, columns=cols, show="headings", height=4,
                                     selectmode="browse", style="Data.Treeview")
        heads = {"name": ("配体", 78), "chain": ("链", 38),
                 "seq": ("编号", 60), "atoms": ("重原子", 58)}
        for key in cols:
            text, width = heads[key]
            self.lig_tree.heading(key, text=text)
            self.lig_tree.column(key, width=width, anchor="center", stretch=True)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.lig_tree.yview)
        self.lig_tree.configure(yscrollcommand=vsb.set)
        self.lig_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.lig_tree.bind("<<TreeviewSelect>>", self._on_ligand_select)
        self.lig_tree.tag_configure("odd", background="#FFFFFF")
        self.lig_tree.tag_configure("even", background="#F8FAF9")
        ttk.Label(parent, textvariable=self.var_ligand_info, style="Muted.TLabel",
                  wraplength=318, justify="left").pack(fill="x", pady=(0, 7))

        ttk.Label(parent, text="判定阈值", style="Sub.TLabel").pack(anchor="w")
        opts = ttk.Frame(parent, style="Card.TFrame")
        opts.pack(fill="x", pady=(4, 0))
        self._build_common_options(opts, include_interface=False)
        ttk.Label(parent, text="疏水仅统计侧链 C/S；金属配位自动识别。",
                  style="Muted.TLabel", wraplength=318, justify="left").pack(
            fill="x", pady=(7, 0))
        self._build_type_legend(parent)

    def _build_pp_tab(self, parent):
        ttk.Label(parent, text="链选择", style="Sub.TLabel").pack(anchor="w")
        grid = ttk.Frame(parent, style="Card.TFrame")
        grid.pack(fill="x", pady=(4, 8))
        ttk.Label(grid, text="链 A", style="Card.TLabel").grid(row=0, column=0,
                                                              sticky="w", pady=2)
        self.cmb_chain_a = ttk.Combobox(grid, textvariable=self.var_chain_a,
                                        state="readonly", width=23)
        self.cmb_chain_a.grid(row=0, column=1, sticky="ew", pady=2, padx=(6, 0))
        ttk.Label(grid, text="链 B", style="Card.TLabel").grid(row=1, column=0,
                                                              sticky="w", pady=2)
        self.cmb_chain_b = ttk.Combobox(grid, textvariable=self.var_chain_b,
                                        state="readonly", width=23)
        self.cmb_chain_b.grid(row=1, column=1, sticky="ew", pady=2, padx=(6, 0))
        ttk.Button(grid, text="交换", style="Small.TButton",
                   command=self._swap_chains).grid(row=0, column=2, rowspan=2,
                                                   padx=(6, 0), sticky="ns")
        grid.columnconfigure(1, weight=1)

        ttk.Label(parent, text="判定阈值", style="Sub.TLabel").pack(anchor="w")
        opts = ttk.Frame(parent, style="Card.TFrame")
        opts.pack(fill="x", pady=(4, 0))
        self._build_common_options(opts, include_interface=True)
        ttk.Label(parent, text="界面残基按重原子距离判定；双侧以两栏网络展示。",
                  style="Muted.TLabel", wraplength=318, justify="left").pack(
            fill="x", pady=(7, 0))
        self._build_type_legend(parent)

    def _build_common_options(self, parent, include_interface: bool):
        parent.columnconfigure(1, weight=1)
        row = 0
        if include_interface:
            ttk.Label(parent, text="界面距离（Å）", style="Card.TLabel").grid(
                row=row, column=0, sticky="w", pady=2)
            ttk.Spinbox(parent, from_=3.0, to=6.0, increment=0.1, width=7,
                        textvariable=self.var_interface_dist).grid(
                row=row, column=1, sticky="w", pady=2)
            row += 1
        for label, var, lo, hi, inc in (
            ("氢键距离（Å）", self.var_hbond_dist, 2.5, 4.5, 0.1),
            ("氢键角度（°）", self.var_hbond_angle, 60, 180, 5),
            ("疏水距离（Å）", self.var_hydro_dist, 3.0, 5.0, 0.1),
        ):
            ttk.Label(parent, text=label, style="Card.TLabel").grid(
                row=row, column=0, sticky="w", pady=2)
            ttk.Spinbox(parent, from_=lo, to=hi, increment=inc, width=7,
                        textvariable=var).grid(row=row, column=1, sticky="w", pady=2)
            row += 1
        checks = ttk.Frame(parent, style="Card.TFrame")
        checks.grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(checks, text="检测 π–π 堆积", variable=self.var_pi,
                        style="Card.TCheckbutton").pack(side="left")
        ttk.Checkbutton(checks, text="标注距离", variable=self.var_show_dist,
                        style="Card.TCheckbutton").pack(side="left", padx=(12, 0))

    def _build_type_legend(self, parent):
        box = ttk.Frame(parent, style="Card.TFrame")
        box.pack(fill="x", pady=(10, 0))
        ttk.Label(box, text="作用类型", style="Sub.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 3))
        items = [
            ("hbond", "氢键"), ("salt", "盐桥"), ("pi", "π–π 堆积"),
            ("metal", "配位键"), ("hydrophobic", "疏水接触"),
        ]
        for idx, (key, label) in enumerate(items):
            cell = ttk.Frame(box, style="Card.TFrame")
            cell.grid(row=1 + idx // 2, column=idx % 2, sticky="w",
                      padx=(0, 14), pady=1)
            dot = tk.Canvas(cell, width=8, height=8, bg=CARD,
                            highlightthickness=0)
            dot.create_oval(1, 1, 7, 7, fill=KIND_COLOR[key], outline="")
            dot.pack(side="left")
            ttk.Label(cell, text=label, style="Muted.TLabel").pack(
                side="left", padx=(3, 0))

    # ------------------------------------------------------------------
    # 主图卡片
    # ------------------------------------------------------------------
    def _build_figure_card(self, parent):
        outer = tk.Frame(parent, bg=CARD, highlightbackground=BORDER,
                         highlightthickness=1)
        parent.add(outer, weight=3)
        inner = ttk.Frame(outer, style="Card.TFrame")
        inner.pack(fill="both", expand=True)

        header = ttk.Frame(inner, style="Card.TFrame", padding=(14, 9, 12, 5))
        header.pack(fill="x")
        bar = tk.Frame(header, bg=PRIMARY, width=3, height=15)
        bar.pack(side="left", padx=(0, 7))
        ttk.Label(header, text="相互作用图", style="Section.TLabel").pack(side="left")
        ttk.Label(header, text="主图 / 定量图", style="Muted.TLabel").pack(
            side="left", padx=(7, 0))

        ttk.Button(header, text="导出", style="Secondary.TButton",
                   command=self._export_figure).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="100%", style="Small.TButton",
                   command=self._fullsize_preview).pack(side="right", padx=(4, 0))
        ttk.Button(header, text="适应窗口", style="Small.TButton",
                   command=self._refit_preview).pack(side="right")
        self.btn_view_quant = ttk.Button(header, text="定量图", style="View.TButton",
                                         command=lambda: self._set_view("quant"),
                                         state="disabled")
        self.btn_view_quant.pack(side="right", padx=(6, 0))
        self.btn_view_diagram = ttk.Button(header, text="互作图",
                                           style="ViewActive.TButton",
                                           command=lambda: self._set_view("diagram"),
                                           state="disabled")
        self.btn_view_diagram.pack(side="right", padx=(10, 0))

        self.toolbar_frame = ttk.Frame(inner, style="Card.TFrame")
        self.toolbar_frame.pack(fill="x", padx=8)

        fig_wrap = tk.Frame(inner, bg=CARD)
        fig_wrap.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.fig_outer = tk.Canvas(fig_wrap, background="white",
                                   highlightthickness=1,
                                   highlightbackground="#E2EAE5")
        vsb = ttk.Scrollbar(fig_wrap, orient="vertical", command=self.fig_outer.yview)
        hsb = ttk.Scrollbar(fig_wrap, orient="horizontal", command=self.fig_outer.xview)
        self.fig_outer.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.fig_outer.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        fig_wrap.rowconfigure(0, weight=1)
        fig_wrap.columnconfigure(0, weight=1)

        self.fig_inner = tk.Frame(self.fig_outer, background="white")
        self.fig_window = self.fig_outer.create_window((0, 0), window=self.fig_inner,
                                                       anchor="nw")
        self.fig_inner.bind("<Configure>", self._on_fig_inner_configure)
        self.fig_outer.bind("<Configure>", self._on_fig_canvas_configure)

    # ------------------------------------------------------------------
    # 结果面板
    # ------------------------------------------------------------------
    def _build_results_card(self, parent):
        outer = tk.Frame(parent, bg=CARD, highlightbackground=BORDER,
                         highlightthickness=1)
        parent.add(outer, weight=1)
        inner = ttk.Frame(outer, style="Card.TFrame", padding=(12, 9, 12, 8))
        inner.pack(fill="both", expand=True)

        head = ttk.Frame(inner, style="Card.TFrame")
        head.pack(fill="x")
        bar = tk.Frame(head, bg=PRIMARY, width=3, height=14)
        bar.pack(side="left", padx=(0, 7))
        ttk.Label(head, text="分析结果", style="Section.TLabel").pack(side="left")
        ttk.Label(head, text="定性分类 + 定量统计", style="Muted.TLabel").pack(
            side="left", padx=(7, 0))
        self.btn_report_top = ttk.Button(head, text="PDF 报告", style="Report.TButton",
                                         command=self._export_report, state="disabled")
        self.btn_report_top.pack(side="right", padx=(0, 6))
        self.btn_csv_top = ttk.Button(head, text="导出 CSV", style="Ghost.TButton",
                                      command=self._export_csv, state="disabled")
        self.btn_csv_top.pack(side="right")

        self._build_kpi(inner)

        nb = ttk.Notebook(inner, style="Result.TNotebook")
        nb.pack(fill="both", expand=True)
        self.result_nb = nb
        table_tab = ttk.Frame(nb, style="Card.TFrame", padding=(2, 4, 2, 2))
        nb.add(table_tab, text=" 定量明细 ")
        self.tree = self._make_table(table_tab)
        residue_tab = ttk.Frame(nb, style="Card.TFrame", padding=(2, 4, 2, 2))
        nb.add(residue_tab, text=" 残基汇总 ")
        self.res_tree = self._make_table(residue_tab)
        summary_tab = ttk.Frame(nb, style="Card.TFrame", padding=(2, 4, 2, 2))
        nb.add(summary_tab, text=" 分析摘要 ")
        self.summary_text = tk.Text(summary_tab, wrap="word", font=(UI_FONT, 9),
                                    background=CARD, foreground="#2B3A32",
                                    relief="flat", padx=8, pady=6, height=5)
        sv = ttk.Scrollbar(summary_tab, orient="vertical",
                           command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=sv.set)
        self.summary_text.pack(side="left", fill="both", expand=True)
        sv.pack(side="right", fill="y")
        self.summary_text.tag_configure("head", font=(UI_FONT, 9, "bold"),
                                        foreground=PRIMARY_DARK, spacing1=4)
        self.summary_text.configure(state="disabled")

        for tree in (self.tree, self.res_tree):
            tree.tag_configure("hbond", foreground=KIND_COLOR["hbond"])
            tree.tag_configure("salt", foreground=KIND_COLOR["salt"])
            tree.tag_configure("pi", foreground=KIND_COLOR["pi"])
            tree.tag_configure("metal", foreground=KIND_COLOR["metal"])
            tree.tag_configure("hydrophobic", foreground=KIND_COLOR["hydrophobic"])
            tree.tag_configure("even", background="#FFFFFF")
            tree.tag_configure("odd", background="#F8FAF9")
        self._set_interaction_columns("pl")
        self._set_residue_columns()

    def _build_kpi(self, parent):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 7))
        specs = [
            ("hbond", "氢键", KIND_COLOR["hbond"]),
            ("salt", "盐桥", KIND_COLOR["salt"]),
            ("pi", "π–π", KIND_COLOR["pi"]),
            ("metal", "配位", KIND_COLOR["metal"]),
            ("hydrophobic", "疏水", KIND_COLOR["hydrophobic"]),
            ("residues", "参与残基", PRIMARY),
        ]
        for key, label, color in specs:
            chip = tk.Frame(row, bg=SOFT, highlightbackground=BORDER,
                            highlightthickness=1)
            chip.pack(side="left", fill="x", expand=True, padx=(0, 6))
            num = tk.Label(chip, text="—", bg=SOFT, fg=color,
                           font=(UI_FONT, 14, "bold"))
            num.pack(pady=(3, 0))
            lbl = tk.Label(chip, text=label, bg=SOFT, fg=MUTED,
                           font=(UI_FONT, 7))
            lbl.pack(pady=(0, 3))
            self.kpi_labels[key] = num
            if key == "residues":
                self.kpi_res_label = lbl

    def _make_table(self, parent) -> ttk.Treeview:
        wrap = ttk.Frame(parent, style="Card.TFrame")
        wrap.pack(fill="both", expand=True)
        tree = ttk.Treeview(wrap, show="headings", height=5, style="Data.Treeview")
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(wrap, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        return tree

    def _set_interaction_columns(self, mode: str):
        if mode == "pl":
            columns = ["类型", "性质", "残基", "蛋白原子", "配体原子",
                       "距离(Å)", "角度(°)", "接触数"]
            widths = [66, 108, 126, 74, 74, 66, 62, 58]
        else:
            columns = ["类型", "性质", "残基(链A)", "原子(链A)", "残基(链B)",
                       "原子(链B)", "距离(Å)", "角度(°)", "接触数"]
            widths = [62, 108, 100, 60, 100, 60, 64, 58, 54]
        cols = [f"c{i}" for i in range(len(columns))]
        self.tree.configure(columns=cols)
        for i, (name, width) in enumerate(zip(columns, widths)):
            self.tree.heading(f"c{i}", text=name)
            anchor = "w" if name.startswith("残基") else "center"
            self.tree.column(f"c{i}", width=width, minwidth=50, anchor=anchor,
                             stretch=True)

    def _set_residue_columns(self):
        columns = [("残基", 118), ("氢键", 48), ("盐桥", 48), ("π–π", 48),
                   ("配位", 48), ("疏水", 48), ("合计", 48), ("最短距离(Å)", 84)]
        cols = [f"r{i}" for i in range(len(columns))]
        self.res_tree.configure(columns=cols)
        for i, (name, width) in enumerate(columns):
            self.res_tree.heading(f"r{i}", text=name)
            anchor = "w" if i == 0 else "center"
            self.res_tree.column(f"r{i}", width=width, minwidth=40, anchor=anchor,
                                 stretch=True)

    # ------------------------------------------------------------------
    # 文件载入
    # ------------------------------------------------------------------
    def _open_dialog(self):
        path = filedialog.askopenfilename(
            title="选择结构文件",
            filetypes=[
                ("结构文件（PDB / mmCIF）",
                 "*.pdb *.ent *.cif *.mmcif *.pdb.gz *.cif.gz *.mmcif.gz"),
                ("PDB 文件", "*.pdb *.ent *.pdb.gz"),
                ("mmCIF 文件", "*.cif *.mmcif *.cif.gz"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            self._load_path(path)

    def _sample_menu_popup(self):
        menu = tk.Menu(self.root, tearoff=False)
        for key, (path, label) in SAMPLES.items():
            menu.add_command(label=label, command=lambda k=key: self._load_sample(k))
        try:
            menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            menu.grab_release()

    def _load_sample(self, key: str):
        entry = SAMPLES.get(key)
        if entry is None or not entry[0].exists():
            messagebox.showwarning("示例缺失", f"未找到示例文件：{entry[0] if entry else key}")
            return
        self._load_path(str(entry[0]))

    def _load_path(self, path: str, auto: bool = True):
        self._set_status("正在读取结构文件…")
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            structure = parse_structure(path)
        except Exception as exc:
            self._set_status("读取失败")
            messagebox.showerror("无法读取结构", f"文件解析失败：\n{exc}")
            return
        finally:
            self.root.config(cursor="")

        self.structure = structure
        self.result = None
        self.last_mode = None
        self.view_mode = "diagram"
        self._set_result_buttons(False)
        self._clear_results()
        self.var_file_state.set("● 已载入")
        self.lbl_file_state.configure(fg=PRIMARY)

        self.lbl_title.configure(text=structure.title[:42]
                                 + ("…" if len(structure.title) > 42 else ""))
        chains = [c for c in structure.chains() if structure.chain_residues(c, True)]
        fmt = "mmCIF" if path.lower().endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz")) else "PDB"
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
        except OSError:
            size_mb = 0.0
        self.lbl_meta.configure(
            text=f"{fmt} · {structure.pdb_id} · {len(chains)} 条蛋白链 · "
                 f"{len(structure.residues)} 残基 · {len(structure.atoms)} 原子 · "
                 f"{size_mb:.1f} MB\n{os.path.basename(path)}"
        )

        self._populate_ligands()
        self._populate_chains()
        if structure.ligands():
            self.nb_mode.select(0)
        else:
            self.nb_mode.select(1)
        self._set_status(f"已载入 {structure.pdb_id}：{len(structure.atoms)} 个原子。")
        if auto:
            self._run_analysis(silent=True)

    def _populate_ligands(self):
        for item in self.lig_tree.get_children():
            self.lig_tree.delete(item)
        self.ligand_map.clear()
        if not self.structure:
            return
        ligands = self.structure.ligands()
        for idx, lig in enumerate(ligands):
            iid = f"lig{idx}"
            self.ligand_map[iid] = lig
            self.lig_tree.insert("", "end", iid=iid, values=(
                lig.resname, lig.chain.strip() or "-",
                f"{lig.resseq}{lig.icode.strip()}", len(lig.heavy_atoms),
            ), tags=("odd" if idx % 2 == 0 else "even",))
        if ligands:
            self.lig_tree.selection_set("lig0")
            self.lig_tree.focus("lig0")
            self.lig_tree.see("lig0")
            self._on_ligand_select()
        else:
            self.var_ligand_info.set("未发现非水 HETATM 配体。")

    def _populate_chains(self):
        if not self.structure:
            return
        chains = [c for c in self.structure.chains()
                  if self.structure.chain_residues(c, True)]
        display_values = []
        self.chain_display_map.clear()
        for chain in chains:
            display = (f"{chain.strip() or '(空白)'}   "
                       f"（{len(self.structure.chain_residues(chain, True))} 残基）")
            self.chain_display_map[display] = chain
            display_values.append(display)
        self.cmb_chain_a.configure(values=display_values)
        self.cmb_chain_b.configure(values=display_values)
        if len(display_values) >= 2:
            a, b = self._default_chain_pair(chains)
            self.var_chain_a.set(self._display_for_chain(a))
            self.var_chain_b.set(self._display_for_chain(b))
        elif len(display_values) == 1:
            self.var_chain_a.set(display_values[0])
            self.var_chain_b.set("")
        else:
            self.var_chain_a.set("")
            self.var_chain_b.set("")

    def _display_for_chain(self, chain: str) -> str:
        for display, value in self.chain_display_map.items():
            if value == chain:
                return display
        return ""

    def _default_chain_pair(self, chains: list[str]) -> tuple[str, str]:
        centroids = {}
        for chain in chains:
            residues = self.structure.chain_residues(chain, True)
            coords = [a.coord for r in residues for a in r.heavy_atoms]
            if coords:
                centroids[chain] = np.mean(np.array(coords), axis=0)
        best = None
        for c1, c2 in combinations(centroids, 2):
            d = float(np.linalg.norm(centroids[c1] - centroids[c2]))
            if best is None or d < best[0]:
                best = (d, c1, c2)
        if best:
            return best[1], best[2]
        return chains[0], chains[1]

    def _swap_chains(self):
        a, b = self.var_chain_a.get(), self.var_chain_b.get()
        self.var_chain_a.set(b)
        self.var_chain_b.set(a)

    def _on_ligand_select(self, _event=None):
        selection = self.lig_tree.selection()
        if not selection:
            return
        lig = self.ligand_map.get(selection[0])
        if lig is None:
            return
        self.var_ligand_info.set(
            f"已选择：{lig.resname} {lig.resseq}{lig.icode.strip()}（链 "
            f"{lig.chain.strip() or '-'}） · {len(lig.heavy_atoms)} 个重原子"
        )

    def _on_mode_changed(self, _event=None):
        self.var_mode.set(self.nb_mode.index(self.nb_mode.select()))

    def _current_mode(self) -> str:
        return "pl" if self.nb_mode.index(self.nb_mode.select()) == 0 else "pp"

    # ------------------------------------------------------------------
    # 分析
    # ------------------------------------------------------------------
    def _read_params(self) -> InteractionParams:
        try:
            hb = float(self.var_hbond_dist.get())
            ang = float(self.var_hbond_angle.get())
            hy = float(self.var_hydro_dist.get())
            itf = float(self.var_interface_dist.get())
        except (tk.TclError, ValueError):
            raise ValueError("参数必须是数字，请检查阈值设置。")
        if not (2.0 <= hb <= 5.0):
            raise ValueError("氢键距离建议在 2.0–5.0 Å 之间。")
        if not (30 <= ang <= 180):
            raise ValueError("氢键角度应在 30–180° 之间。")
        if not (2.5 <= hy <= 6.0):
            raise ValueError("疏水距离建议在 2.5–6.0 Å 之间。")
        if not (2.5 <= itf <= 8.0):
            raise ValueError("界面距离建议在 2.5–8.0 Å 之间。")
        return InteractionParams(
            hbond_dist=hb, hbond_angle=ang, hydrophobic_dist=hy,
            interface_dist=itf, pi_enabled=bool(self.var_pi.get()),
        )

    def _run_analysis(self, silent: bool = False):
        if self.structure is None:
            if not silent:
                messagebox.showinfo("请先打开文件", "请先打开一个 PDB / mmCIF 结构文件。")
            return
        try:
            params = self._read_params()
            mode = self._current_mode()
            self._set_status("正在分析相互作用…")
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            if mode == "pl":
                selection = self.lig_tree.selection()
                if not selection:
                    raise ValueError("请先在“配体选择”列表中选择一个配体。")
                ligand = self.ligand_map.get(selection[0])
                if ligand is None:
                    raise ValueError("未能读取所选配体。")
                result = analyze_pl(self.structure, ligand, params)
            else:
                chain_a = self.chain_display_map.get(self.var_chain_a.get())
                chain_b = self.chain_display_map.get(self.var_chain_b.get())
                if not chain_a or not chain_b:
                    raise ValueError("请选择两条蛋白链。")
                if chain_a == chain_b:
                    raise ValueError("请选择两条不同的链。")
                result = analyze_pp(self.structure, chain_a, chain_b, params)

            self.result = result
            self.last_mode = mode
            self.last_show_distances = bool(self.var_show_dist.get())
            self.view_mode = "diagram"
            fig = self._build_current_view(1.0)
            fig = self._fit_figure_for_preview(fig)
            self._show_figure(fig)
            self._fill_table(result, mode)
            self._fill_residue_table(result)
            self._fill_summary(result.summary)
            self._update_kpis(result, mode)
            self._set_result_buttons(True)
            self._update_view_buttons()

            s = result.stats
            if mode == "pl":
                metal_txt = f" · 配位键 {s['metal']}" if s.get("metal") else ""
                self._set_status(
                    f"分析完成：{len(result.nodes)} 个口袋残基 · 氢键 {s['hbond']} · "
                    f"盐桥 {s['salt']} · π–π {s['pi']}{metal_txt} · 疏水 {s['hydrophobic']}"
                )
            else:
                self._set_status(
                    f"分析完成：界面残基 {s['n_iface_a']} + {s['n_iface_b']} · "
                    f"氢键 {s['hbond']} · 盐桥 {s['salt']} · π–π {s['pi']} · "
                    f"疏水 {s['hydrophobic']}"
                )
        except Exception as exc:
            self._set_status("分析失败")
            if silent:
                self._clear_preview_with_message(str(exc))
            else:
                messagebox.showerror("分析失败", str(exc))
                traceback.print_exc()
        finally:
            self.root.config(cursor="")

    # ------------------------------------------------------------------
    # 图与视图
    # ------------------------------------------------------------------
    def _build_current_view(self, scale: float) -> Figure | None:
        if self.result is None or self.last_mode is None:
            return None
        if self.view_mode == "quant":
            return build_quant_figure(self.result, scale=scale)
        if self.last_mode == "pl":
            return build_pl_figure(self.result, scale=scale,
                                   show_distances=self.last_show_distances)
        return build_pp_figure(self.result, scale=scale,
                               show_distances=self.last_show_distances)

    def _fit_figure_for_preview(self, fig: Figure) -> Figure:
        self.root.update_idletasks()
        try:
            dpr = self.root.winfo_fpixels("1i") / 96.0
        except tk.TclError:
            dpr = 1.0
        avail_w = max(self.fig_outer.winfo_width() - 28, 420)
        avail_h = max(self.fig_outer.winfo_height() - 24, 320)
        px_w = fig.get_figwidth() * 110.0 * dpr
        px_h = fig.get_figheight() * 110.0 * dpr
        scale = min(1.0, avail_w / px_w, avail_h / px_h)
        scale = max(scale, 0.64)
        if scale >= 0.97:
            return fig
        rebuilt = self._build_current_view(scale)
        return rebuilt if rebuilt is not None else fig

    def _set_view(self, mode: str):
        if self.result is None:
            return
        if mode == self.view_mode:
            return
        self.view_mode = mode
        self._update_view_buttons()
        fig = self._build_current_view(1.0)
        fig = self._fit_figure_for_preview(fig)
        self._show_figure(fig)
        if mode == "quant":
            self._set_status("定量图：残基互作谱 + 相互作用距离分布。")
        else:
            self._set_status("互作图：相互作用二维示意图。")

    def _update_view_buttons(self):
        has_result = self.result is not None
        self.btn_view_diagram.configure(
            style="ViewActive.TButton" if self.view_mode == "diagram" else "View.TButton",
            state="normal" if has_result else "disabled",
        )
        self.btn_view_quant.configure(
            style="ViewActive.TButton" if self.view_mode == "quant" else "View.TButton",
            state="normal" if has_result else "disabled",
        )

    def _set_result_buttons(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for btn in (self.btn_export_png, self.btn_export_csv, self.btn_csv_top,
                    self.btn_report_top):
            btn.configure(state=state)
        if not enabled:
            self._update_view_buttons()

    def _refit_preview(self):
        fig = self._build_current_view(1.0)
        if fig is None:
            return
        self._show_figure(self._fit_figure_for_preview(fig))

    def _fullsize_preview(self):
        fig = self._build_current_view(1.0)
        if fig is None:
            return
        self._show_figure(fig)

    def _clear_figure_widgets(self):
        if self.toolbar is not None:
            self.toolbar.destroy()
            self.toolbar = None
        self.fig_canvas = None
        for child in list(self.fig_inner.winfo_children()):
            try:
                child.destroy()
            except tk.TclError:
                pass

    def _show_placeholder(self):
        if self.result is not None:
            return
        self._clear_figure_widgets()
        holder = tk.Frame(self.fig_inner, bg="white", height=420)
        holder.pack(fill="both", expand=True)
        holder.pack_propagate(False)
        tk.Label(holder, text="打开结构文件，开始相互作用分析",
                 bg="white", fg="#8B978F", font=(UI_FONT, 13)).pack(pady=(120, 4))
        tk.Label(holder, text="支持 PDB / mmCIF，自动识别配体或界面链；也可从左侧载入示例。",
                 bg="white", fg="#AAB5AE", font=(UI_FONT, 9)).pack()

    def _clear_preview_with_message(self, message: str):
        self._clear_figure_widgets()
        holder = tk.Frame(self.fig_inner, bg="white", height=380)
        holder.pack(fill="both", expand=True)
        holder.pack_propagate(False)
        tk.Label(holder, text="未能自动生成示意图", bg="white", fg="#8B978F",
                 font=(UI_FONT, 11)).pack(pady=(110, 4))
        tk.Label(holder, text=message, bg="white", fg="#AAB5AE",
                 font=(UI_FONT, 9), wraplength=520, justify="center").pack()

    def _show_figure(self, fig: Figure):
        self._clear_figure_widgets()
        fig.set_dpi(110)
        canvas = FigureCanvasTkAgg(fig, master=self.fig_inner)
        widget = canvas.get_tk_widget()
        widget.pack()
        canvas.draw()
        self.fig_canvas = canvas
        self.toolbar = ChineseToolbar(canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x")
        self.fig_inner.update_idletasks()
        self.fig_outer.configure(scrollregion=self.fig_outer.bbox("all"))
        self.fig_outer.xview_moveto(0)
        self.fig_outer.yview_moveto(0)

    def _on_fig_inner_configure(self, _event=None):
        self.fig_outer.configure(scrollregion=self.fig_outer.bbox("all"))

    def _on_fig_canvas_configure(self, event):
        width = max(event.width, self.fig_inner.winfo_reqwidth())
        self.fig_outer.itemconfigure(self.fig_window, width=width)

    def _set_initial_sash(self):
        try:
            height = self.right_paned.winfo_height()
            if height > 300:
                self.right_paned.sashpos(0, int(height * 0.68))
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # 表格与摘要
    # ------------------------------------------------------------------
    def _fill_table(self, result, mode: str):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._set_interaction_columns(mode)
        for idx, row in enumerate(result.table_rows):
            kind_tag = TAG_BY_KIND_CN.get(row[0], "")
            parity = "odd" if idx % 2 == 0 else "even"
            tags = (kind_tag, parity) if kind_tag else (parity,)
            self.tree.insert("", "end", values=row, tags=tags)

    def _fill_residue_table(self, result):
        for item in self.res_tree.get_children():
            self.res_tree.delete(item)
        self._set_residue_columns()
        for idx, row in enumerate(result.residue_rows):
            parity = "odd" if idx % 2 == 0 else "even"
            self.res_tree.insert("", "end", values=row, tags=(parity,))

    def _fill_summary(self, text: str):
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        heads = ("结构文件：", "配体：", "分析链：", "界面残基：",
                 "参与相互作用的残基：", "结合口袋残基：")
        for line in text.splitlines():
            is_head = line.endswith("：") or line.startswith(heads)
            self.summary_text.insert("end", line + "\n", ("head",) if is_head else ())
        self.summary_text.configure(state="disabled")

    def _update_kpis(self, result, mode: str):
        stats = result.stats
        for key in ("hbond", "salt", "pi", "metal", "hydrophobic"):
            self.kpi_labels[key].configure(text=str(stats.get(key, 0)))
        if mode == "pl":
            count = len(result.nodes)
            self.kpi_res_label.configure(text="结合残基")
        else:
            count = stats.get("n_iface_a", 0) + stats.get("n_iface_b", 0)
            self.kpi_res_label.configure(text="界面残基")
        self.kpi_labels["residues"].configure(text=str(count))

    def _clear_results(self):
        for tree in (self.tree, self.res_tree):
            for item in tree.get_children():
                tree.delete(item)
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.configure(state="disabled")
        for label in self.kpi_labels.values():
            label.configure(text="—")
        self.kpi_res_label.configure(text="参与残基")
        self._show_placeholder()
        self._update_view_buttons()
        self._set_result_buttons(False)

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def _export_figure(self):
        if self.result is None or self.last_mode is None:
            messagebox.showinfo("暂无结果", "请先完成一次分析。")
            return
        default = self._default_export_name("png")
        path = filedialog.asksaveasfilename(
            title="导出当前图", defaultextension=".png", initialfile=default,
            filetypes=[("PNG 位图（600 dpi）", "*.png"),
                       ("PDF 矢量图", "*.pdf"),
                       ("SVG 矢量图", "*.svg")],
        )
        if not path:
            return
        self._set_status("正在导出图片…")
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            fig = self._build_current_view(1.0)
            fig.savefig(path, dpi=600, facecolor="white")
            self._set_status(f"图片已导出：{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", f"图片导出失败：\n{exc}")
            traceback.print_exc()
        finally:
            self.root.config(cursor="")

    def _export_csv(self):
        if self.result is None:
            messagebox.showinfo("暂无结果", "请先完成一次分析。")
            return
        path = filedialog.asksaveasfilename(
            title="导出数据表", defaultextension=".csv",
            initialfile=self._default_export_name("csv"),
            filetypes=[("CSV 表格", "*.csv"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            interaction_cols = [self.tree.heading(c, "text")
                                for c in self.tree["columns"]]
            residue_cols = ["残基", "氢键", "盐桥", "π–π", "配位",
                            "疏水", "合计", "最短距离(Å)"]
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.writer(fh)
                writer.writerow(["PDB", self.structure.pdb_id if self.structure else ""])
                if isinstance(self.result, PLResult):
                    lig = self.result.ligand
                    writer.writerow(["配体", lig.resname, lig.resseq,
                                     "链", lig.chain.strip() or ""])
                else:
                    writer.writerow(["链A", self.result.chain_a.strip(),
                                     "链B", self.result.chain_b.strip()])
                writer.writerow([])
                writer.writerow(["相互作用明细"])
                writer.writerow(interaction_cols)
                for iid in self.tree.get_children():
                    writer.writerow(self.tree.item(iid, "values"))
                writer.writerow([])
                writer.writerow(["残基汇总"])
                writer.writerow(residue_cols)
                for iid in self.res_tree.get_children():
                    writer.writerow(self.res_tree.item(iid, "values"))
            self._set_status(f"数据表已导出：{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", f"数据表导出失败：\n{exc}")

    def _export_report(self):
        if self.result is None:
            messagebox.showinfo("暂无结果", "请先完成一次分析。")
            return
        path = filedialog.asksaveasfilename(
            title="生成中英双语 PDF 报告", defaultextension=".pdf",
            initialfile=self._default_report_name(),
            filetypes=[("PDF 报告", "*.pdf")],
        )
        if not path:
            return
        win = tk.Toplevel(self.root)
        win.title("生成 PDF 报告")
        win.geometry("460x150")
        win.resizable(False, False)
        win.transient(self.root)
        win.configure(background=CARD)
        win.protocol("WM_DELETE_WINDOW", lambda: None)
        tk.Label(win, text="正在生成中英双语报告", bg=CARD, fg=INK,
                 font=(UI_FONT, 11, "bold")).pack(pady=(20, 4))
        msg_var = tk.StringVar(value="准备…")
        tk.Label(win, textvariable=msg_var, bg=CARD, fg=MUTED,
                 font=(UI_FONT, 8)).pack()
        bar = ttk.Progressbar(win, mode="indeterminate", length=400)
        bar.pack(pady=12)
        bar.start(12)
        try:
            win.grab_set()
        except tk.TclError:
            pass
        progress = lambda msg: (msg_var.set(msg), win.update_idletasks(), win.update())
        self.root.config(cursor="watch")
        try:
            from .report import generate_report
            generate_report(self.result, path, progress=progress)
        except Exception as exc:
            traceback.print_exc()
            try:
                win.grab_release()
            except tk.TclError:
                pass
            win.destroy()
            messagebox.showerror("报告生成失败", f"PDF 报告生成失败：\n{exc}")
            self._set_status("报告生成失败")
        else:
            try:
                win.grab_release()
            except tk.TclError:
                pass
            bar.stop()
            win.destroy()
            self._set_status(f"PDF 报告已生成：{path}")
            if messagebox.askyesno("报告已生成",
                                   f"PDF 报告已生成：\n{path}\n\n是否立即打开？"):
                try:
                    os.startfile(path)
                except OSError:
                    pass
        finally:
            self.root.config(cursor="")

    def _default_report_name(self) -> str:
        if self.structure is None:
            return "interaction_report.pdf"
        pdb = self.structure.pdb_id
        if self.last_mode == "pl" and isinstance(self.result, PLResult):
            lig = self.result.ligand
            tag = f"{lig.resname}{lig.resseq}{lig.chain.strip()}".replace(" ", "")
        elif self.last_mode == "pp" and isinstance(self.result, PPResult):
            tag = f"{self.result.chain_a.strip()}_{self.result.chain_b.strip()}"
        else:
            tag = "analysis"
        return f"{pdb}_{tag}_report.pdf"

    def _default_export_name(self, ext: str) -> str:
        if self.structure is None:
            return f"interactions.{ext}"
        pdb = self.structure.pdb_id
        if self.last_mode == "pl" and isinstance(self.result, PLResult):
            lig = self.result.ligand
            tag = f"{lig.resname}{lig.resseq}{lig.chain.strip()}".replace(" ", "")
        elif self.last_mode == "pp" and isinstance(self.result, PPResult):
            tag = f"{self.result.chain_a.strip()}_{self.result.chain_b.strip()}"
        else:
            tag = "analysis"
        if ext == "csv":
            return f"{pdb}_{tag}_interactions.csv"
        suffix = "quant" if self.view_mode == "quant" else "diagram"
        return f"{pdb}_{tag}_{suffix}.{ext}"

    # ------------------------------------------------------------------
    # 其他
    # ------------------------------------------------------------------
    def _set_status(self, message: str):
        self.status_var.set(message)

    def _show_help(self):
        text = (
            "使用步骤\n"
            "1. 点击“打开结构文件…”选择 .pdb/.ent/.cif/.mmcif（支持 gzip），\n"
            "   或从“示例 ▾”载入 1A4W（PDB / mmCIF）与 1BRS（蛋白–蛋白）。\n"
            "2. 蛋白–小分子：在配体列表中选择配体，默认选中重原子最多的非水 HETATM 残基。\n"
            "3. 蛋白–蛋白：指定两条不同的链，默认给出空间上最接近的一对链。\n"
            "4. 调整阈值后点击“分析并绘图”。\n\n"
            "判定规则（阈值可调）\n"
            "• 氢键：供体–受体重原子距离 ≤ 设定值，并检查 D–H…A 角度（虚拟氢）。\n"
            "• 疏水：蛋白侧链与配体/另一条链的 C/S 原子距离 ≤ 设定值。\n"
            "• 盐桥：正负带电基团距离 ≤ 4.0 Å，按残基对合并。\n"
            "• π–π：芳环质心距离 ≤ 5.5 Å，区分平行堆积与 T 型。\n"
            "• 配位：配体金属与蛋白 N/O/S 距离 ≤ 2.9 Å；配位原子不重复计为氢键。\n\n"
            "定性 + 定量\n"
            "• “定量明细”给出作用性质、原子对、距离、角度和接触数。\n"
            "• “残基汇总”按残基统计五类作用数量与最短距离。\n"
            "• KPI 指标条与“定量图”提供残基互作谱和距离分布，可直接用于论文。\n\n"
            "图形与导出\n"
            "• 互作图：绿色虚线为氢键，紫色为盐桥，棕色为金属配位，\n"
            "  红色“睫毛”弧为疏水残基；蛋白–蛋白为双栏界面网络。\n"
            "• 图内文字全部为英文；界面为中文。\n"
            "• PNG 600 dpi，PDF/SVG 为矢量；CSV 同时包含明细与残基汇总。\n"
            "• “生成中英双语 PDF 报告”会输出封面、摘要、15 张出版级图件\n"
            "  （主图/定量/指纹/组成/BSA/几何/接触/热点/富集/网络/类型-类别/\n"
            "  氢键联合几何/距离统计/埋藏空间/仪表盘）、关键表格、方法与参考文献；\n"
            "  每张图均有中英文图注与解读，字体已嵌入。\n"
            "• 报告中的 SASA/BSA、配体描述符与界面组成需要数秒计算。\n"
            "• 预览支持工具栏缩放/平移，也可“适应窗口 / 100%”。\n\n"
            "数据说明\n"
            "• 仅读取第一个 MODEL；水分子不参与统计。\n"
            "• 疏水接触仅统计侧链 C/S；相互作用结果仅供研究参考。"
        )
        self._show_text_dialog("使用说明", text)

    def _show_about(self):
        text = (
            f"LigPlot-CN {__version__}\n\n"
            "参考 LigPlot+ 的表达方式、独立实现的蛋白相互作用可视化与定量分析工具。\n\n"
            "功能：\n"
            "• 蛋白–小分子 / 蛋白–蛋白两种模式\n"
            "• 氢键、疏水、盐桥、π–π、金属配位\n"
            "• 定性性质标注 + 残基级定量汇总 + 出版级定量图\n"
            "• SASA/BSA、配体描述符、界面组成与热点评价\n"
            "• CNS 风格中英双语 PDF 报告（15 张图，字体嵌入）\n"
            "• PDB / mmCIF（gzip）\n"
            "• PNG 600 dpi / PDF / SVG / CSV\n"
            "• 中文界面，英文图注\n\n"
            "实现：Python 3 + NumPy + Matplotlib + Tkinter\n"
            "本软件不包含 LigPlot+ 的任何源代码。"
        )
        self._show_text_dialog("关于 LigPlot-CN", text)

    def _show_text_dialog(self, title: str, text: str):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("600x560")
        win.transient(self.root)
        win.configure(background=CARD)
        frame = ttk.Frame(win, style="Card.TFrame", padding=14)
        frame.pack(fill="both", expand=True)
        txt = tk.Text(frame, wrap="word", font=(UI_FONT, 9), background=CARD,
                      foreground="#2B3A32", relief="flat", padx=8, pady=6)
        sb = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        txt.insert("1.0", text)
        txt.configure(state="disabled")
        ttk.Button(win, text="关闭", style="Primary.TButton",
                   command=win.destroy).pack(pady=(0, 12))


def main():
    _enable_dpi_awareness()
    root = tk.Tk()
    app = LigPlotApp(root)
    try:
        root.state("zoomed")
    except tk.TclError:
        pass
    root.after(250, lambda: _apply_dark_titlebar(root))
    root.mainloop()


if __name__ == "__main__":
    main()
