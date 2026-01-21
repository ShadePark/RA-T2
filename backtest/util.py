#!/usr/bin/env python
# coding: utf-8
"""Backtest core for Static / Dynamic / Total (combined) portfolios.

This module is a refactor of `prj2_backtest_1.py` so it can be imported
without immediately running backtests.

Usage (Python):
    from prj2_backtest_core import run_static, run_dynamic, run_total

Notes:
- Requires: FinanceDataReader, pandas, numpy, matplotlib
- Data is fetched from online sources via FinanceDataReader.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)


# -------------------------
# Generic helpers
# -------------------------

def get_close(ticker: str, start: str, end: Optional[str] = None) -> pd.Series:
    """Fetch Close prices."""
    return fdr.DataReader(ticker, start, end)["Close"]


def get_adj_close(ticker: str, start: str, end: Optional[str] = None) -> pd.Series:
    """Fetch Adjusted Close prices."""
    return fdr.DataReader(ticker, start, end)["Adj Close"]


def calc_daily_return(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Daily returns."""
    return prices.pct_change().fillna(0)


def calc_cum_return(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Cumulative return (growth of $1)."""
    return prices / prices.iloc[0]


def evaluate_performance(
    daily_ret: pd.Series,
    cum_ret: pd.Series,
    risk_free_rate: float = 0.02,
    enable_print: bool = True,
) -> Tuple[float, pd.Series, float, float, float]:
    """Return (CAGR, drawdown_series, MDD, Sharpe, annual_vol)."""
    total_ret = float(cum_ret.iloc[-1])
    years = len(cum_ret) / 252
    cagr = total_ret ** (1 / years) - 1

    historical_max = cum_ret.cummax()
    dd = (cum_ret - historical_max) / historical_max * 100
    mdd = float(dd.min())

    daily_vol = float(daily_ret.std())
    annual_vol = daily_vol * np.sqrt(252)

    annual_return = float(daily_ret.mean()) * 252
    sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol != 0 else np.nan

    if enable_print:
        print(f"최종 수익률   : {(cum_ret.iloc[-1]-1)*100:.2f}%")
        print(f"수익성 (CAGR) : {cagr*100:.2f}%")
        print(f"안정성 (MDD)  : {mdd:.2f}%")
        print(f"연간 변동성   : {annual_vol*100:.2f}%")
        print(f"샤프 지수     : {sharpe_ratio:.4f}")

    return cagr, dd, mdd, sharpe_ratio, annual_vol


def get_rebalancing_dates(indexed_data: pd.DataFrame | pd.Series, period: str = "month") -> pd.DatetimeIndex:
    """리밸런싱 날짜 추출 (각 기간의 첫 영업일)."""
    freq_map = {"month": "ME", "quarter": "QE", "year": "YE"}
    if period not in freq_map:
        raise ValueError("period must be 'month', 'quarter', or 'year'")
    freq = freq_map[period]

    if isinstance(indexed_data, pd.Series):
        base = indexed_data.to_frame("x")
    else:
        base = indexed_data

    rebalancing_dates = base.groupby(pd.Grouper(freq=freq)).apply(lambda x: x.index[0])  # 첫번째 날짜
    # return pd.DatetimeIndex(rebalancing_dates.sort_index())
    return rebalancing_dates.sort_index()


def plot_portfolio_result(
    tickers: Iterable[str],
    daily_rets: Optional[pd.DataFrame],
    port_cum_ret: pd.Series,
    dd: pd.Series,
    mdd: float,
    title_suffix: str = "",
    *,
    show: bool = True,
):
    """Plot cumulative return + drawdown.

    Notes
    -----
    - In GUI mode, call with show=False and render the returned Figure via a Canvas.
    - This function keeps the original visual style used in the notebook/pop-up plots.
    """
    from matplotlib.figure import Figure
    import matplotlib.ticker as mticker

    fig = Figure(figsize=(10, 8), dpi=100)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    # (1) cumulative
    ax1.plot(port_cum_ret.index, port_cum_ret.values, label="Portfolio", color="#d62728", linewidth=1.5)

    if daily_rets is not None and tickers is not None:
        start_date = port_cum_ret.index[0]
        end_date = port_cum_ret.index[-1]
        subset_daily_rets = daily_rets.loc[start_date:end_date]

        for t in tickers:
            if t in subset_daily_rets.columns:
                ind_cum = (1 + subset_daily_rets[t]).cumprod()
                ax1.plot(ind_cum.index, ind_cum.values, label=f"{t}", alpha=0.3, linewidth=0.8, linestyle="--")

    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax1.ticklabel_format(style="plain", axis="y")
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    title = f"Portfolio Cumulative Return{(' - ' + title_suffix) if title_suffix else ''}"
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize="small")
    ax1.grid(True, alpha=0.3)

    # (2) drawdown
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.3)
    ax2.plot(dd.index, dd.values, linewidth=0.5)
    ax2.set_title("Portfolio Drawdown (MDD)", fontsize=12)
    ax2.axhline(mdd, color="red", linestyle="--", label=f"Max DD: {mdd:.2f}%")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Drawdown (%)")

    fig.tight_layout()

    if show:
        import matplotlib.pyplot as plt
        # Render in an interactive window
        plt.figure(figsize=(10, 8))
        canvas = plt.gcf().canvas
        # Replace current figure with our Figure for display
        # (matplotlib doesn't let us "show" a Figure instance directly without a backend canvas)
        # So in scripts/notebook, prefer creating plots directly; this branch is kept for backward compatibility.
        plt.close(plt.gcf())
        # Fallback: show via pyplot by drawing again using pyplot
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(port_cum_ret.index, port_cum_ret, label="Portfolio", color="#d62728", linewidth=1.5)
        if daily_rets is not None and tickers is not None:
            subset_daily_rets = daily_rets.loc[port_cum_ret.index[0]:port_cum_ret.index[-1]]
            for t in tickers:
                if t in subset_daily_rets.columns:
                    ind_cum = (1 + subset_daily_rets[t]).cumprod()
                    plt.plot(ind_cum.index, ind_cum, label=f"{t}", alpha=0.3, linewidth=0.8, linestyle="--")
        plt.yscale("log")
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax.ticklabel_format(style="plain", axis="y")
        plt.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="upper left", fontsize="small")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.fill_between(dd.index, dd, 0, alpha=0.3)
        plt.plot(dd.index, dd, linewidth=0.5)
        plt.title("Portfolio Drawdown (MDD)", fontsize=12)
        plt.axhline(mdd, color="red", linestyle="--", label=f"Max DD: {mdd:.2f}%")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.ylabel("Drawdown (%)")
        plt.tight_layout()
        plt.show()

    return fig



def cal_rebalancing_portfolio(
    close_data: pd.DataFrame,
    period: str = "month",
    term: int = 1,
    weight_df: Optional[pd.DataFrame] = None,
    enable_plot: bool = True,
    title_suffix: str = "",
) -> Tuple[pd.Series, pd.Series]:
    """Chunk-based rebalancing backtest (same logic as notebook)."""
    rebal_dates = get_rebalancing_dates(close_data, period)

    if weight_df is not None:
        rebal_dates = rebal_dates[rebal_dates.isin(weight_df.index)]
        if rebal_dates.empty:
            raise ValueError("close_data와 weight_df의 리밸런싱 날짜가 일치하지 않습니다.")

    if weight_df is None:
        n_assets = len(close_data.columns)
        weight_df = pd.DataFrame(index=rebal_dates, columns=close_data.columns, data=1 / n_assets)

    first_date = rebal_dates[0]

    full_daily_rets = close_data.pct_change().fillna(0)
    daily_rets = full_daily_rets.loc[first_date:]

    portfolio_chunks = []
    total_value = 1.0

    full_dates = list(rebal_dates)
    if full_dates[-1] < daily_rets.index[-1]:
        full_dates.append(daily_rets.index[-1])

    for start, end in zip(full_dates[:-term:term], full_dates[term::term]):
        if start not in weight_df.index:
            continue

        current_weights = weight_df.loc[start]
        chunk_rets = daily_rets.loc[start:end].iloc[1:]
        if chunk_rets.empty:
            continue

        cum_growth = (1 + chunk_rets).cumprod()
        chunk_value = (cum_growth * current_weights).sum(axis=1) * total_value
        portfolio_chunks.append(chunk_value)
        total_value = float(chunk_value.iloc[-1])

    if not portfolio_chunks:
        raise ValueError("백테스트 구간이 비어 있습니다. 날짜/데이터를 확인하세요.")

    portfolio_cum_ret = pd.concat(portfolio_chunks)
    portfolio_cum_ret.loc[first_date] = 1.0
    portfolio_cum_ret = portfolio_cum_ret.sort_index()

    portfolio_day_ret = portfolio_cum_ret.pct_change().fillna(0)

    if enable_plot:
        historical_max = portfolio_cum_ret.cummax()
        dd = (portfolio_cum_ret - historical_max) / historical_max * 100
        mdd = float(dd.min())

        tickers = [c for c in weight_df.columns if c in close_data.columns]
        if "cash" in tickers:
            tickers.remove("cash")
        plot_daily_rets = full_daily_rets.loc[portfolio_cum_ret.index]
        plot_portfolio_result(tickers, plot_daily_rets, portfolio_cum_ret, dd, mdd, title_suffix=title_suffix)

    return portfolio_day_ret, portfolio_cum_ret


# -------------------------
# Portfolio presets
# -------------------------

STATIC_WEIGHT: Dict[str, float] = {
    "GLD": 0.25,
    "QQQ": 0.25,
    "IVV": 0.161,
    "VGLT": 0.129,
    "SIVR": 0.03,
    "DBC": 0.03,
    "VEU": 0.03,
    "EMB": 0.03,
    "TIP": 0.03,
    "LQD": 0.03,
    "VGIT": 0.03,
}

DYNAMIC_COLS = ['GLD', 'BIL', 'XLE', 'SPY', 'QQQ']


@dataclass
class BacktestResult:
    daily_ret: pd.Series
    cum_ret: pd.Series
    cagr: float
    mdd: float
    sharpe: float
    vol: float
    plot_df: pd.DataFrame | None = None

def make_portfolio_figure_from_plot_df(
    plot_df: pd.DataFrame,
    title_suffix: str = "",
    *,
    portfolio_label: str = "Portfolio",
    show_assets: bool = True,
) -> "matplotlib.figure.Figure":
    """
    Build a Figure that matches the original pop-up plot style.

    Parameters
    ----------
    plot_df : DataFrame
        Must contain 'PORT' and optionally asset columns (tickers). Values are
        cumulative return series (wealth index) normalized to start at 1.
    """
    from matplotlib.figure import Figure
    import matplotlib.ticker as mticker

    if plot_df is None or plot_df.empty or "PORT" not in plot_df.columns:
        raise ValueError("plot_df must contain a 'PORT' column")

    port = plot_df["PORT"].dropna()
    dd = (port - port.cummax()) / port.cummax() * 100.0
    mdd = float(dd.min()) if not dd.empty else 0.0

    fig = Figure(figsize=(10, 8), dpi=100)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    # (1) cumulative
    ax1.plot(port.index, port.values, label=portfolio_label, color="#d62728", linewidth=1.5)

    if show_assets:
        for c in [c for c in plot_df.columns if c != "PORT"]:
            s = plot_df[c].dropna()
            if s.empty:
                continue
            ax1.plot(s.index, s.values, label=str(c), alpha=0.3, linewidth=0.8, linestyle="--")

    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax1.ticklabel_format(style="plain", axis="y")
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    title = f"Portfolio Cumulative Return{(' - ' + title_suffix) if title_suffix else ''}"
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize="small")
    ax1.grid(True, alpha=0.3)

    # (2) drawdown
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.3)
    ax2.plot(dd.index, dd.values, linewidth=0.5)
    ax2.set_title("Portfolio Drawdown (MDD)", fontsize=12)
    ax2.axhline(mdd, color="red", linestyle="--", label=f"Max DD: {mdd:.2f}%")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Drawdown (%)")

    fig.tight_layout()
    return fig




def run_static(
    start: str = "2010-01-04",
    end: str = "2026",
    rebal_period: str = "month",
    rebal_term: int = 4,
    enable_plot: bool = True,
    enable_print: bool = True,
) -> BacktestResult:
    """Run static portfolio backtest."""
    static_cols = list(STATIC_WEIGHT.keys())

    close_data = pd.DataFrame()
    for ticker in static_cols:
        s = get_adj_close(ticker, start, end)
        close_data = pd.concat([close_data, s], axis=1)
    close_data.columns = static_cols
    close_data.index = pd.to_datetime(close_data.index)

    rebal_date = get_rebalancing_dates(close_data, rebal_period)
    static_weight_df = pd.DataFrame([list(STATIC_WEIGHT.values())] * len(rebal_date), index=rebal_date, columns=static_cols)

    day_ret, cum_ret = cal_rebalancing_portfolio(
        close_data=close_data,
        period=rebal_period,
        term=rebal_term,
        weight_df=static_weight_df,
        enable_plot=enable_plot,
        title_suffix=f"Static (term={rebal_term})",
    )
    cagr, dd, mdd, sharpe, vol = evaluate_performance(day_ret, cum_ret, enable_print=enable_print)

    # plot_df: portfolio (PORT) + each asset cumulative (normalized)
    asset_cum = close_data / close_data.iloc[0]
    plot_df = asset_cum.copy()
    plot_df.insert(0, "PORT", cum_ret.reindex(asset_cum.index))

    return BacktestResult(day_ret, cum_ret, cagr, mdd, sharpe, vol, plot_df=plot_df)


def _build_dynamic_weights(close_data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Build dynamic weight dataframe (logic 그대로)."""
    TRD_1M = 21
    TRD_3M = 63

    W_RISK_OFF_DEF = {'GLD': 0.5, 'BIL': 0.5}
    W_RISK_OFF_INF = {'BIL': 0.7, 'XLE': 0.3}
    W_RECOVERY_DEF = {'BIL': 1.0}
    W_RECOVERY_INF = {'GLD': 0.6, 'BIL': 0.2, 'SPY': 0.1, 'XLE': 0.1}
    W_RISK_ON_WEAK = {'GLD': 0.7, 'SPY': 0.3}
    W_RISK_ON_STRONG = {'GLD': 0.3, 'QQQ': 0.7}

    spy = close_data["SPY"]
    spy_ma200 = spy.rolling(200).mean()
    spy_mom6m = spy.pct_change(126)
    spy_ret_1m = spy.pct_change(TRD_1M)
    spy_ret_3m = spy.pct_change(TRD_3M)

    tlt = get_close("TLT", start, end)
    tlt_ret_3m = tlt.pct_change(TRD_3M)

    qqq = close_data["QQQ"]
    qqq_mom6m = qqq.pct_change(126)

    vix = get_close("^VIX", start, end)

    def_shock = (spy_ret_3m < 0) & (tlt_ret_3m > 0)
    inf_shock = (spy_ret_3m < 0) & (tlt_ret_3m < -0.08)

    trend_on = spy > spy_ma200
    mom_on = spy_mom6m > 0

    rel_str = qqq_mom6m > spy_mom6m
    vix_low = vix < 18
    ma_slope = spy_ma200 > spy_ma200.shift(20)

    ind = (
        pd.DataFrame(
            {
                "vix": vix,
                "trend_on": trend_on,
                "mom_on": mom_on,
                "rel_str": rel_str,
                "vix_low": vix_low,
                "ma_slope": ma_slope,
                "def_shock": def_shock,
                "inf_shock": inf_shock,
            }
        )
        .dropna()
    )
    ind = ind.loc['2010-01-04':'2026']

    ind_date = get_rebalancing_dates(ind)

    dynamic_weight_df = pd.DataFrame()
    state = "RISK_OFF"
    risk_type = "DEF"

    for d in ind_date:
        row = ind.loc[d]

        if row["vix"] > 30 or not (row["trend_on"] and row["mom_on"]):
            state = "RISK_OFF"
            risk_type = "INF" if row["inf_shock"] else "DEF"
            dynamic_weight = W_RISK_OFF_INF if row["inf_shock"] else W_RISK_OFF_DEF
        else:
            if state == "RISK_OFF" or (state == "RECOVERY" and row["vix"] > 25):
                state = "RECOVERY"
                dynamic_weight = W_RECOVERY_INF if risk_type == "INF" else W_RECOVERY_DEF
            else:
                state = "RISK_ON"
                strength = int(row["rel_str"]) + int(row["vix_low"]) + int(row["ma_slope"])
                dynamic_weight = W_RISK_ON_WEAK if strength <= 1 else W_RISK_ON_STRONG

        w_df = pd.DataFrame([dynamic_weight], columns=DYNAMIC_COLS, index=[d]).fillna(0.0)
        dynamic_weight_df = pd.concat([dynamic_weight_df, w_df], axis=0)

    return dynamic_weight_df


def run_dynamic(
    start: str = "2010-01-04",
    end: str = "2026",
    rebal_period: str = "month",
    rebal_term: int = 1,
    enable_plot: bool = True,
    enable_print: bool = True,
) -> BacktestResult:
    """Run dynamic portfolio backtest."""
    close_data = pd.DataFrame()
    for ticker in DYNAMIC_COLS:
        s = get_adj_close(ticker, '2009', '2026')
        close_data = pd.concat([close_data, s], axis=1)
    close_data.columns = DYNAMIC_COLS
    close_data.index = pd.to_datetime(close_data.index)

    dynamic_weight_df = _build_dynamic_weights(close_data, '2009', '2026')

    day_ret, cum_ret = cal_rebalancing_portfolio(
        close_data=close_data,
        period=rebal_period,
        term=rebal_term,
        weight_df=dynamic_weight_df,
        enable_plot=enable_plot,
        title_suffix=f"Dynamic (term={rebal_term})",
    )
    cagr, dd, mdd, sharpe, vol = evaluate_performance(day_ret, cum_ret, enable_print=enable_print)

    # plot_df: portfolio (PORT) + each asset cumulative (normalized)
    asset_cum = close_data / close_data.iloc[0]
    plot_df = asset_cum.copy()
    plot_df.insert(0, "PORT", cum_ret.reindex(asset_cum.index))

    return BacktestResult(day_ret, cum_ret, cagr, mdd, sharpe, vol, plot_df=plot_df)


def run_total(
    s_ratio: float = 0.7,
    d_ratio: float = 0.3,
    total_asset: float = 10_000.0,
    start: str = "2010-01-04",
    end: str = "2026",
    static_rebal_term_months: int = 1,
    enable_plot: bool = True,
    enable_print: bool = True,
) -> Tuple[pd.DataFrame, BacktestResult, BacktestResult, BacktestResult]:
    """Run combined portfolio backtest (with fees/slippage and cash tracking).

    Returns:
        total_portfolio_df, static_result, dynamic_result, total_result

    total_portfolio_df columns: Static, Dynamic, Total
    """
    if not (0 <= s_ratio <= 1 and 0 <= d_ratio <= 1):
        raise ValueError("s_ratio/d_ratio must be within [0,1].")
    if abs((s_ratio + d_ratio) - 1.0) > 1e-9:
        raise ValueError("s_ratio + d_ratio must equal 1.0")

    static_cols = list(STATIC_WEIGHT.keys())
    all_cols = sorted(list(set(static_cols) | set(DYNAMIC_COLS)))

    close_data = pd.DataFrame()
    for ticker in all_cols:
        s = get_adj_close(ticker, "2009", end)
        close_data = pd.concat([close_data, s], axis=1)
    close_data.columns = all_cols
    close_data.index = pd.to_datetime(close_data.index)

    # Rebalancing anchors: last trading day of each month (DatetimeIndex)
    rebal_date = get_rebalancing_dates(close_data.loc[start:end])
    # print(rebal_date)

    # static weights per rebal date
    static_weight_df = pd.DataFrame(
        [list(STATIC_WEIGHT.values())] * len(rebal_date),
        index=rebal_date,
        columns=static_cols,
    )

    # dynamic weights
    dyn_close = close_data[DYNAMIC_COLS]
    dynamic_weight_df = _build_dynamic_weights(dyn_close, start="2009", end=end)
    # align dynamic weights to rebal_date (intersection)
    dynamic_weight_df = dynamic_weight_df.loc[dynamic_weight_df.index.intersection(rebal_date)]

    # fees
    broker_fee = 0.0025
    tax_rate = 0.0
    slippage_rate = 0.001

    s_cash = total_asset * s_ratio
    d_cash = total_asset * d_ratio

    def rebalancing_process(
        price_data: pd.DataFrame,
        weight: pd.Series,
        prev_quantity: pd.Series,
        cash: float,
        is_rebal: bool,
    ):
        price_on_rebal_day = price_data.iloc[0]
        asset = float((prev_quantity * price_on_rebal_day).sum() + cash)

        capacity = (asset * (1 - (broker_fee + tax_rate + slippage_rate) * 2)) * weight

        if is_rebal:
            quantity = (capacity // price_on_rebal_day).astype(float)
            diff_quantity = quantity - prev_quantity
            trade_amount = (diff_quantity.abs() * price_on_rebal_day).astype(float)

            total_brokerage = float((trade_amount * broker_fee).sum())
            sell_amount = trade_amount[diff_quantity < 0]
            total_tax = float((sell_amount * tax_rate).sum())
            total_slippage = float((trade_amount * slippage_rate).sum())
            total_cost = total_brokerage + total_tax + total_slippage

            cash = float(asset - float((quantity * price_on_rebal_day).sum()) - total_cost)
        else:
            quantity = prev_quantity

        asset_flow = quantity * price_data
        asset_flow["cash"] = round(cash, 2)
        return asset_flow, quantity, cash

    # init
    month_count = 0

    s_data = close_data[static_cols]
    d_data = close_data[DYNAMIC_COLS]

    s_portfolio = pd.DataFrame()
    d_portfolio = pd.DataFrame()

    s_prev_quantity = pd.Series(0.0, index=static_cols)
    d_prev_quantity = pd.Series(0.0, index=DYNAMIC_COLS)

    # Build monthly segments [period_start, period_end] from anchor dates.
    # (DatetimeIndex has no .items())
    anchors = list(pd.DatetimeIndex(rebal_date).sort_values())
    if not anchors:
        raise ValueError("리밸런싱 날짜가 비어 있습니다. start/end를 확인하세요.")

    # Ensure the last segment reaches the last available date in the selected window
    last_date = close_data.loc[start:end].index.max()
    if pd.isna(last_date):
        raise ValueError("가격 데이터가 비어 있습니다. 티커/기간을 확인하세요.")
    if anchors[-1] < last_date:
        anchors.append(last_date)

    # for period_start, period_end in zip(anchors[:-1], anchors[1:]):
    for period_end, period_start in rebal_date.items():
        # static
        s_slice = s_data.loc[period_start:period_end]
        # print(f'{period_start}-{period_end}')
        if len(s_slice) < 2:
            continue
        s_asset_flow, s_prev_quantity, s_cash = rebalancing_process(
            price_data=s_slice,
            weight=static_weight_df.loc[period_start],
            prev_quantity=s_prev_quantity,
            cash=s_cash,
            is_rebal=(month_count % static_rebal_term_months == 0),
        )
        s_portfolio = pd.concat([s_portfolio, s_asset_flow])

        # dynamic
        # (if dynamic_weight_df missing that date, fallback to all-cash BIL)
        if period_start in dynamic_weight_df.index:
            d_w = dynamic_weight_df.loc[period_start]
        else:
            d_w = pd.Series({"BIL": 1.0}, index=DYNAMIC_COLS).fillna(0.0)

        d_slice = d_data.loc[period_start:period_end]
        if len(d_slice) < 2:
            month_count += 1
            continue
        d_asset_flow, d_prev_quantity, d_cash = rebalancing_process(
            price_data=d_slice,
            weight=d_w,
            prev_quantity=d_prev_quantity,
            cash=d_cash,
            is_rebal=True,
        )
        d_portfolio = pd.concat([d_portfolio, d_asset_flow])

        month_count += 1

    s_sum = s_portfolio.sum(axis=1)
    d_sum = d_portfolio.sum(axis=1)

    total_portfolio = pd.concat([s_sum, d_sum], axis=1)
    total_portfolio.columns = ["Static", "Dynamic"]
    total_portfolio["Total"] = total_portfolio["Static"] + total_portfolio["Dynamic"]

    total_cum_ret = total_portfolio / total_portfolio.iloc[0]
    total_day_ret = total_cum_ret.pct_change().fillna(0)
    
    # results
    if enable_print:
        print("== 정적 포트폴리오 ==")
    s_cagr, s_dd, s_mdd, s_sharpe, s_vol = evaluate_performance(total_day_ret["Static"], total_cum_ret["Static"], enable_print=enable_print)

    if enable_print:
        print("== 동적 포트폴리오 ==")
    d_cagr, d_dd, d_mdd, d_sharpe, d_vol = evaluate_performance(total_day_ret["Dynamic"], total_cum_ret["Dynamic"], enable_print=enable_print)

    if enable_print:
        print("== 종합 포트폴리오 ==")
    t_cagr, t_dd, t_mdd, t_sharpe, t_vol = evaluate_performance(total_day_ret["Total"], total_cum_ret["Total"], enable_print=enable_print)

    # plotting
    if enable_plot:
        if enable_print:
            pass
        plot_portfolio_result(["Static", "Dynamic"], total_day_ret[["Static", "Dynamic"]], total_cum_ret["Total"], t_dd, t_mdd, title_suffix=f"Total (S={s_ratio:.0%}, D={d_ratio:.0%})")

        total_portfolio[["Static", "Dynamic", "Total"]].plot(figsize=(10, 4))
        plt.title("Portfolio Value (Static / Dynamic / Total)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    static_res = BacktestResult(total_day_ret["Static"], total_cum_ret["Static"], s_cagr, s_mdd, s_sharpe, s_vol)
    dynamic_res = BacktestResult(total_day_ret["Dynamic"], total_cum_ret["Dynamic"], d_cagr, d_mdd, d_sharpe, d_vol)
    total_res = BacktestResult(total_day_ret["Total"], total_cum_ret["Total"], t_cagr, t_mdd, t_sharpe, t_vol)

    # plot_df for GUI: Total as PORT (solid), Static/Dynamic as dashed components
    try:
        total_res.plot_df = pd.DataFrame({
            "PORT": total_cum_ret["Total"].dropna(),
            "Static": total_cum_ret["Static"].reindex(total_cum_ret.index).dropna(),
            "Dynamic": total_cum_ret["Dynamic"].reindex(total_cum_ret.index).dropna(),
        }).dropna(how="any")
    except Exception:
        total_res.plot_df = None


    return total_portfolio[["Static", "Dynamic", "Total"]], static_res, dynamic_res, total_res

