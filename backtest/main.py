#!/usr/bin/env python
# coding: utf-8
"""Tkinter GUI runner (tabs) with embedded matplotlib plots.

Run:
    python prj2_gui.py

Features:
- Select and run Static / Dynamic / Total portfolios.
- Results shown in tabs (Static/Dynamic/Total).
- Each tab shows:
  - Embedded plot (cumulative return + drawdown)
  - Metrics table (per run)
  - Save CSV / Save PNG
  - Overlay multiple runs on the same plot
"""

from __future__ import annotations

GUI_VERSION = "v7"

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Matplotlib (embedded canvas) - MUST run on the main thread only.
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import pandas as pd

from util import run_static, run_dynamic, run_total, BacktestResult, make_portfolio_figure_from_plot_df


# -----------------------------
# Matplotlib font (Korean-safe)
# -----------------------------
def _set_korean_font():
    """Best-effort Korean font setup to avoid glyph warnings."""
    candidates = [
        "Malgun Gothic",   # Windows
        "AppleGothic",     # macOS
        "NanumGothic",     # Linux (common)
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]

    chosen = None
    for name in candidates:
        try:
            _ = font_manager.findfont(name, fallback_to_default=False)
            chosen = name
            break
        except Exception:
            continue

    if not chosen:
        # Scan installed fonts
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in installed:
                chosen = name
                break

    if chosen:
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = [chosen]
        rcParams["axes.unicode_minus"] = False


_set_korean_font()


def _drawdown_from_cum(cum: pd.Series) -> pd.Series:
    """Drawdown (%) from cumulative series (wealth index)."""
    hist = cum.cummax()
    dd = (cum - hist) / hist * 100.0
    return dd


def _total_return_pct(cum: pd.Series) -> float:
    if cum.empty:
        return 0.0
    return float((cum.iloc[-1] - 1.0) * 100.0)


@dataclass
class RunRecord:
    run_id: str
    label: str
    when: datetime
    res: BacktestResult
    extra_df: pd.DataFrame | None = None  # for Total tab (Static/Dynamic/Total columns)


class ResultTab(ttk.Frame):
    """One tab: plot + metrics table + save buttons."""

    def __init__(self, parent: ttk.Notebook, tab_key: str, title: str):
        super().__init__(parent)
        self.tab_key = tab_key
        self.title = title

        self.runs: list[RunRecord] = []
        self.var_overlay = tk.BooleanVar(value=True)

        self._fig: Figure | None = None
        self._canvas: FigureCanvasTkAgg | None = None
        self._toolbar: NavigationToolbar2Tk | None = None

        self._build()

    def _build(self):
        # layout: plot (top) + table/buttons (bottom)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=6)  # plot (taller)
        self.rowconfigure(1, weight=2)  # table

        # plot container
        plot_wrap = ttk.Frame(self)
        plot_wrap.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))
        plot_wrap.columnconfigure(0, weight=1)
        plot_wrap.rowconfigure(1, weight=1)

        hdr = ttk.Frame(plot_wrap)
        hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(hdr, text=self.title, font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Checkbutton(hdr, text="여러 실행 결과 누적 표시", variable=self.var_overlay, command=self.redraw).pack(side="right")

        self._fig = Figure(figsize=(9.2, 6.6), dpi=100)
        self._fig.suptitle("Ready", fontsize=12)

        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_wrap)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        self._toolbar = NavigationToolbar2Tk(self._canvas, plot_wrap, pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.grid(row=2, column=0, sticky="ew", pady=(6, 0))

        # bottom container: table + buttons
        bottom = ttk.Frame(self)
        bottom.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(0, weight=1)

        cols = ("run", "tot_ret", "cagr", "mdd", "vol", "sharpe")
        self.tree = ttk.Treeview(bottom, columns=cols, show="headings", height=7)
        self.tree.heading("run", text="Run")
        self.tree.heading("tot_ret", text="누적수익률(%)")
        self.tree.heading("cagr", text="CAGR(%)")
        self.tree.heading("mdd", text="MDD(%)")
        self.tree.heading("sharpe", text="Sharpe")
        self.tree.heading("vol", text="Vol(%)")

        self.tree.column("run", width=170, anchor="w")
        for c in cols[1:]:
            self.tree.column(c, width=100, anchor="e")

        vsb = ttk.Scrollbar(bottom, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        btns = ttk.Frame(bottom)
        btns.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        btns.columnconfigure(0, weight=1)

        left = ttk.Frame(btns)
        left.pack(side="left")
        ttk.Button(left, text="CSV 저장", command=self.save_csv).pack(side="left", padx=(0, 8))
        ttk.Button(left, text="PNG 저장", command=self.save_png).pack(side="left", padx=(0, 8))
        ttk.Button(left, text="기록/그래프 초기화", command=self.clear).pack(side="left")

        ttk.Label(btns, text="(테이블에서 Run 선택 후 저장 가능)", foreground="#666666").pack(side="right")

    # -----------------
    # Run management
    # -----------------
    def add_run(self, label: str, res: BacktestResult, extra_df: pd.DataFrame | None = None):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec = RunRecord(
            run_id=run_id,
            label=label,
            when=datetime.now(),
            res=res,
            extra_df=extra_df,
        )
        self.runs.append(rec)
        self._append_table_row(rec)
        self.redraw(select_last=True)

    def _append_table_row(self, rec: RunRecord):
        tot = _total_return_pct(rec.res.cum_ret.dropna())
        self.tree.insert(
            "",
            "end",
            iid=rec.run_id,
            values=(
                rec.when.strftime("%m-%d %H:%M:%S"),
                f"{tot:,.2f}",
                f"{rec.res.cagr*100:,.2f}",
                f"{rec.res.mdd:,.2f}",
                f"{rec.res.vol*100:,.2f}",
                f"{rec.res.sharpe:,.4f}",
            ),
        )

    def clear(self):
        self.runs.clear()
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self.redraw()

    def _selected_or_latest(self) -> RunRecord | None:
        sel = self.tree.selection()
        if sel:
            iid = sel[0]
            for r in self.runs:
                if r.run_id == iid:
                    return r
        return self.runs[-1] if self.runs else None

    # -----------------
    # Plotting
    # -----------------
    def redraw(self, select_last: bool = False):
        if not self._fig or not self._canvas:
            return

        if select_last and self.runs:
            self.tree.selection_set(self.runs[-1].run_id)

        self._fig.clear()
        self._fig.suptitle(self.title, fontsize=12)

        ax1 = self._fig.add_subplot(2, 1, 1)
        ax2 = self._fig.add_subplot(2, 1, 2, sharex=ax1)

        ax1.set_ylabel("Cumulative")
        ax2.set_ylabel("Drawdown (%)")

        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        overlay = bool(self.var_overlay.get())
        runs_to_plot = self.runs if overlay else (self.runs[-1:] if self.runs else [])

        if not runs_to_plot:
            ax1.text(0.5, 0.5, "No results yet", ha="center", va="center", transform=ax1.transAxes)
            self._fig.tight_layout(rect=(0, 0, 1, 0.95))
            self._canvas.draw()
            return

        if self.tab_key in ("static", "dynamic", "total"):

            # Rebuild the plot in the original "popup" style:
            #  - Portfolio: solid red
            #  - Assets (latest run only): dashed
            #  - Optional overlay: previous portfolios as faint gray solid

            import matplotlib.ticker as mticker

            self._fig.clf()
            ax1 = self._fig.add_subplot(2, 1, 1)
            ax2 = self._fig.add_subplot(2, 1, 2, sharex=ax1)

            if not runs_to_plot:
                ax1.text(0.5, 0.5, "No results yet.", ha="center", va="center", transform=ax1.transAxes)
                ax1.axis("off")
                ax2.axis("off")
                self._fig.tight_layout()
                self._canvas.draw()
                return

            latest = runs_to_plot[-1]
            pdf_latest = getattr(latest.res, "plot_df", None)

            def _get_port_series(rec: RunRecord) -> pd.Series:
                pdf = getattr(rec.res, "plot_df", None)
                if pdf is not None and not pdf.empty and "PORT" in pdf.columns:
                    return pdf["PORT"].dropna()
                return rec.res.cum_ret.dropna()

            # Previous portfolios (overlay)
            if self.var_overlay.get() and len(runs_to_plot) > 1:
                for rec in runs_to_plot[:-1]:
                    port_prev = _get_port_series(rec)
                    if port_prev.empty:
                        continue
                    ax1.plot(port_prev.index, port_prev.values, color="gray", alpha=0.35, linewidth=1.0)

            # Latest portfolio (solid red)
            port_latest = _get_port_series(latest)
            ax1.plot(port_latest.index, port_latest.values, label="Portfolio", color="#d62728", linewidth=1.5)

            # Latest assets (dashed)
            if pdf_latest is not None and not pdf_latest.empty:
                for c in [c for c in pdf_latest.columns if c != "PORT"]:
                    s = pdf_latest[c].dropna()
                    if s.empty:
                        continue
                    s = s.loc['2010-01-04':'2026']
                    ax1.plot(s.index, s.values, label=str(c), alpha=0.3, linewidth=0.8, linestyle="--")

            # Match notebook pop-up formatting
            ax1.set_yscale("log")
            ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax1.yaxis.set_minor_formatter(mticker.ScalarFormatter())
            ax1.ticklabel_format(style="plain", axis="y")
            ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.5)

            title_suffix = latest.label
            title_main = "Portfolio Cumulative Return"
            ax1.set_title(f"{title_main}{(' - ' + title_suffix) if title_suffix else ''}", fontsize=14, fontweight="bold")
            ax1.legend(loc="upper left", fontsize="small")
            ax1.grid(True, alpha=0.3)

            # Drawdown panel (latest)
            dd = _drawdown_from_cum(port_latest) if not port_latest.empty else pd.Series(dtype=float)
            mdd = float(dd.min()) if not dd.empty else 0.0

            ax2.fill_between(dd.index, dd.values, 0, alpha=0.3)
            ax2.plot(dd.index, dd.values, linewidth=0.5)
            ax2.set_title("Portfolio Drawdown (MDD)", fontsize=12)
            ax2.axhline(mdd, color="red", linestyle="--", label=f"Max DD: {mdd:.2f}%")
            ax2.legend(loc="lower right")
            ax2.grid(True, alpha=0.3)
            ax2.set_ylabel("Drawdown (%)")

            self._fig.tight_layout()
            self._canvas.draw()
            return


    def save_csv(self):
        rec = self._selected_or_latest()
        if not rec:
            messagebox.showinfo("저장", "저장할 결과가 없습니다.")
            return

        default = f"{self.tab_key}_{rec.run_id}.csv"
        path = filedialog.asksaveasfilename(
            title="CSV로 저장",
            defaultextension=".csv",
            initialfile=default,
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return

        cum = rec.res.cum_ret.dropna()
        daily = rec.res.daily_ret.dropna()
        df = pd.DataFrame({"daily_ret": daily, "cum_ret": cum})
        df["drawdown_pct"] = _drawdown_from_cum(df["cum_ret"])
        df.to_csv(path, encoding="utf-8-sig")
        messagebox.showinfo("저장", f"CSV 저장 완료:\n{path}")

    def save_png(self):
        if not self._fig:
            return
        rec = self._selected_or_latest()
        run_id = rec.run_id if rec else datetime.now().strftime("%Y%m%d_%H%M%S")
        default = f"{self.tab_key}_{run_id}.png"
        path = filedialog.asksaveasfilename(
            title="PNG로 저장",
            defaultextension=".png",
            initialfile=default,
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        self._fig.savefig(path, dpi=150, bbox_inches="tight")
        messagebox.showinfo("저장", f"PNG 저장 완료:\n{path}")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Project2 Backtest Runner")
        self.geometry("1300x1105")
        self.minsize(1150, 720)

        # queue for worker -> UI messages
        self.q: "queue.Queue[tuple]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_flag = threading.Event()

        # Tk vars (UI thread only)
        self.var_static = tk.BooleanVar(value=True)
        self.var_dynamic = tk.BooleanVar(value=True)
        self.var_total = tk.BooleanVar(value=True)

        self.var_static_term = tk.IntVar(value=1)
        self.var_dynamic_term = tk.IntVar(value=1)

        self.var_sratio = tk.DoubleVar(value=0.7)
        self.var_dratio = tk.DoubleVar(value=0.3)
        self.var_total_asset = tk.DoubleVar(value=10_000.0)

        self.var_print = tk.BooleanVar(value=True)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Debug: show which files are running (helps avoid old-file confusion)
        try:
            import prj2_backtest_core as _core
            self.log(f"GUI={GUI_VERSION}  GUI file={__file__}")
            self.log(f"Core file={_core.__file__}")
        except Exception:
            pass

        # poll queue
        self.after(100, self._poll_queue)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # left controls
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        title = ttk.Label(left, text="Backtest 선택", font=("Segoe UI", 14, "bold"))
        title.grid(row=0, column=0, sticky="w", columnspan=3, **pad)
        title.grid_configure(pady=(0, 10))

        ttk.Checkbutton(left, text="정적 포트폴리오", variable=self.var_static).grid(row=1, column=0, sticky="w", **pad)
        ttk.Label(left, text="리밸런싱 term (개월):").grid(row=1, column=1, sticky="e", **pad)
        ttk.Spinbox(left, from_=1, to=24, textvariable=self.var_static_term, width=6).grid(row=1, column=2, sticky="w", **pad)

        ttk.Checkbutton(left, text="동적 포트폴리오", variable=self.var_dynamic).grid(row=2, column=0, sticky="w", **pad)
        ttk.Label(left, text="리밸런싱 term (개월):").grid(row=2, column=1, sticky="e", **pad)
        ttk.Spinbox(left, from_=1, to=12, textvariable=self.var_dynamic_term, width=6).grid(row=2, column=2, sticky="w", **pad)

        ttk.Checkbutton(left, text="종합 포트폴리오", variable=self.var_total).grid(row=3, column=0, sticky="w", **pad)

        sep = ttk.Separator(left, orient="horizontal")
        sep.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 8), padx=10)

        ttk.Label(left, text="종합 포트폴리오 파라미터", font=("Segoe UI", 12, "bold")).grid(row=5, column=0, sticky="w", columnspan=3, **pad)

        ttk.Label(left, text="정적 비중 (0~1)").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(left, textvariable=self.var_sratio, width=10).grid(row=6, column=1, sticky="w", **pad)

        ttk.Label(left, text="동적 비중 (0~1)").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(left, textvariable=self.var_dratio, width=10).grid(row=7, column=1, sticky="w", **pad)

        ttk.Label(left, text="초기 자본금 (USD)").grid(row=8, column=0, sticky="w", **pad)
        ttk.Entry(left, textvariable=self.var_total_asset, width=12).grid(row=8, column=1, sticky="w", **pad)

        sep2 = ttk.Separator(left, orient="horizontal")
        sep2.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(12, 8), padx=10)

        ttk.Checkbutton(left, text="콘솔 출력", variable=self.var_print).grid(row=10, column=0, sticky="w", **pad)

        self.btn_run = ttk.Button(left, text="실행", command=self.on_run)
        self.btn_run.grid(row=11, column=0, pady=(14, 6), sticky="w", padx=10)

        self.btn_quit = ttk.Button(left, text="종료", command=self.on_close)
        self.btn_quit.grid(row=11, column=1, pady=(14, 6), sticky="w", padx=10)

        self.txt_log = tk.Text(left, height=12, wrap="word")
        self.txt_log.grid(row=12, column=0, columnspan=3, sticky="nsew", padx=10, pady=(8, 10))
        left.rowconfigure(12, weight=1)
        left.columnconfigure(0, weight=1)

        # right: Notebook tabs
        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True, padx=6, pady=6)

        self.tab_static = ResultTab(self.nb, "static", "정적 포트폴리오")
        self.tab_dynamic = ResultTab(self.nb, "dynamic", "동적 포트폴리오")
        self.tab_total = ResultTab(self.nb, "total", "종합 포트폴리오")

        self.nb.add(self.tab_static, text="정적")
        self.nb.add(self.tab_dynamic, text="동적")
        self.nb.add(self.tab_total, text="종합")

        # start with plot area wider
        self.after(50, lambda: self._init_sash(paned, target=360))

    # --------------
    # UI helpers
    # --------------
    
    def _init_sash(self, paned: ttk.Panedwindow, target: int = 360, tries: int = 30):
        """Initialize PanedWindow sash position reliably after first layout.
        Tk sometimes reports width=1 early; retry a few times.
        """
        try:
            w = paned.winfo_width()
        except Exception:
            w = 0

        if w and w > target + 50:
            try:
                paned.sashpos(0, target)
            except Exception:
                pass
            return

        if tries <= 0:
            # last attempt
            try:
                paned.sashpos(0, target)
            except Exception:
                pass
            return

        # retry shortly
        self.after(50, lambda: self._init_sash(paned, target=target, tries=tries - 1))

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_log.insert("end", f"[{ts}] {msg}\n")
        self.txt_log.see("end")

    def set_running(self, running: bool):
        self.btn_run.configure(state=("disabled" if running else "normal"))

    # --------------
    # Run
    # --------------
    def on_run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("실행 중", "이미 실행 중입니다.")
            return

        do_static = bool(self.var_static.get())
        do_dynamic = bool(self.var_dynamic.get())
        do_total = bool(self.var_total.get())

        if not (do_static or do_dynamic or do_total):
            messagebox.showinfo("선택 필요", "실행할 포트폴리오를 선택하세요.")
            return

        # snapshot params (UI thread)
        params = dict(
            do_static=do_static,
            do_dynamic=do_dynamic,
            do_total=do_total,
            static_term=int(self.var_static_term.get()),
            dynamic_term=int(self.var_dynamic_term.get()),
            s_ratio=float(self.var_sratio.get()),
            d_ratio=float(self.var_dratio.get()),
            total_asset=float(self.var_total_asset.get()),
            enable_print=bool(self.var_print.get()),
        )

        self.set_running(True)
        self.log("실행 시작...")

        self.stop_flag.clear()
        self.worker = threading.Thread(target=self._worker_run, args=(params,), daemon=True)
        self.worker.start()

    def _worker_run(self, params: dict):
        try:
            enable_print = params["enable_print"]

            if params["do_static"]:
                self.q.put(("log", "정적 포트폴리오 실행 중..."))
                res = run_static(rebal_term=params["static_term"], enable_plot=False, enable_print=enable_print)
                self.q.put(("result", "static", res, None))

            if params["do_dynamic"]:
                self.q.put(("log", "동적 포트폴리오 실행 중..."))
                res = run_dynamic(rebal_term=params["dynamic_term"], enable_plot=False, enable_print=enable_print)
                self.q.put(("result", "dynamic", res, None))

            if params["do_total"]:
                self.q.put(("log", "종합 포트폴리오 실행 중..."))
                total_df, static_res, dynamic_res, total_res = run_total(
                    s_ratio=params["s_ratio"],
                    d_ratio=params["d_ratio"],
                    total_asset=params["total_asset"],
                    static_rebal_term_months=params["static_term"],
                    enable_plot=False,
                    enable_print=enable_print,
                )
                self.q.put(("result_total", total_res, total_df))

            self.q.put(("done",))
        except Exception as e:
            self.q.put(("error", str(e)))

    # --------------
    # Queue polling
    # --------------
    def _poll_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                kind = item[0]

                if kind == "log":
                    self.log(item[1])

                elif kind == "result":
                    _, key, res, extra = item
                    if key == "static":
                        self.tab_static.add_run("정적", res, extra_df=None)
                        self.nb.select(self.tab_static)
                        self._print_summary("정적", res)
                    elif key == "dynamic":
                        self.tab_dynamic.add_run("동적", res, extra_df=None)
                        self.nb.select(self.tab_dynamic)
                        self._print_summary("동적", res)

                elif kind == "result_total":
                    _, total_res, total_df = item
                    # Add only to the Total tab (plot uses df to show components)
                    self.tab_total.add_run("종합", total_res, extra_df=total_df)
                    self.nb.select(self.tab_total)
                    self._print_summary("종합", total_res)

                elif kind == "error":
                    self.log(f"에러: {item[1]}")
                    messagebox.showerror("실행 에러", item[1])
                    self.set_running(False)

                elif kind == "done":
                    self.log("실행 완료.")
                    self.set_running(False)

        except queue.Empty:
            pass
        finally:
            self.after(120, self._poll_queue)

    def _print_summary(self, label: str, res: BacktestResult):
        cum = res.cum_ret.dropna()
        self.log(f"{label} 최종 수익률   : {_total_return_pct(cum):.2f}%")
        self.log(f"{label} 수익성 (CAGR) : {res.cagr*100:.2f}%")
        self.log(f"{label} 안정성 (MDD)  : {res.mdd*100:.2f}%")
        self.log(f"{label} 연간 변동성   : {res.vol*100:.2f}%")
        self.log(f"{label} 샤프 지수     : {res.sharpe:.4f}")

    def on_close(self):
        try:
            self.stop_flag.set()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
