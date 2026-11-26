"""Strategy & Analytics Engine for TradingPro v3.1 (Early-Cycle & Macro-Liquidity Edition).

Implements:
- Classic Supertrend (and optional legacy Divergence-Adaptive mode)
- Early-Cycle Fresh-Flip + Anti-Chase Cap (Close vs SMA50 <= max_ext_pct)
- Causal Macro-Liquidity Filter (Fed 3M Rate Trend ^IRX, 10Y Yield ^TNX, US Dollar DXY)
- Optional Weekly-Supertrend Multi-Timeframe Filter (off by default to prevent late chasing)
- Realistic Next-Day Open execution (with optional legacy Same-Bar Close mode)
- Risk-adjusted metrics (CAGR, MaxDD, Sharpe, Calmar, Profit Factor)
- 63-day volatility-adjusted Momentum (Relative Strength) score
- ATR/Stop-based Position Sizing Calculator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


DEFAULT_STRATEGY_CONFIG: Dict[str, object] = {
    "config_version": 3,
    "atr_len": 10,
    "base_factor": 3.0,
    "macro_filter_mode": "fed_plus_qqq50",  # "fed_plus_qqq50" | "fed_pause" | "strict_liq" | "off"
    "anti_chase_max_ext_pct": 25.0,         # Max % above SMA50 on fresh entry (0 = disabled)
    "fresh_flip_bars": 3,               # Fresh Daily-Supertrend flip window in trading days
    "use_weekly_filter": False,         # Off by default (avoids late-chasing tops)
    "use_div_adaptive": False,          # Pure classic Supertrend by default
    "div_sens": 0.3,
    "piv_len": 2,
    "osc_len": 6,
    "osc_smooth": 7,
    "div_bars": 200,
    "exec_next_open": True,
    "fee_pct": 0.10,                    # 0.10% per side
    "slippage_pct": 0.05,               # 0.05% per side
    "breadth_threshold": 35.0,
    "portfolio_eur": 25000.0,
    "risk_per_trade_pct": 1.0,
    "max_position_pct": 15.0,
}


@dataclass
class StrategyResult:
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    df_indicators: pd.DataFrame
    trades: List[Dict[str, object]]
    current_signal: str          # "BUY", "WAIT", "EXTENDED", "SELL"
    daily_regime: int            # 1 or -1
    weekly_regime: int           # 1 or -1
    macro_ok: int                # 1 or 0
    signal_date: str
    last_close: float
    stop_price: float
    stop_dist_pct: float
    ext50_pct: float
    mom_score: float
    atr_pct: float
    rsi_14: float
    strat_return: float
    bh_return: float
    strat_cagr: float
    strat_dd: float
    bh_dd: float
    sharpe: float
    calmar: float
    profit_factor: float
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    num_trades: int
    report_text: str = ""
    extra: Dict[str, object] = field(default_factory=dict)


def calculate_osc(df: pd.DataFrame, length: int, smooth: int) -> pd.Series:
    body = (df["Close"] - df["Open"]).abs()
    avg_body = body.rolling(window=length).mean()
    direction = np.where(
        df["Close"] > df["Open"], 1.0, np.where(df["Close"] < df["Open"], -1.0, 0.0)
    )
    weight = np.zeros_like(avg_body, dtype=float)
    mask = (avg_body != 0) & (~np.isnan(avg_body))
    weight[mask] = body.values[mask] / avg_body.values[mask]
    contrib = direction * weight
    rolling = pd.Series(contrib, index=df.index).rolling(window=length).sum()
    norm = rolling / (length * 2) * 100.0
    return norm.ewm(span=smooth, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, length: int) -> pd.Series:
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / max(1, int(length)), adjust=False).mean()


def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ema_up = up.ewm(alpha=1.0 / length, adjust=False).mean()
    ema_down = down.ewm(alpha=1.0 / length, adjust=False).mean()
    rs = ema_up / ema_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calculate_max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    running_max = series.cummax()
    dd = (series - running_max) / running_max.replace(0.0, np.nan)
    val = dd.min()
    return float(val * 100.0) if pd.notna(val) else 0.0


def calculate_drawdown_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    running_max = series.cummax()
    return ((series - running_max) / running_max.replace(0.0, np.nan)).fillna(0.0) * 100.0


def get_detailed_stats(values: List[float]) -> Tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    s = pd.Series(values, dtype=float)
    return float(s.mean()), float(s.median()), float(s.quantile(0.25)), float(s.quantile(0.75))


def calculate_weekly_supertrend_regime(
    df: pd.DataFrame, atr_len: int = 10, factor: float = 3.0
) -> Tuple[pd.Series, int]:
    """Computes strictly causal Weekly Supertrend regime aligned to daily index."""
    wdf = (
        df.resample("W-FRI")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    if len(wdf) < 15:
        ones = pd.Series(1, index=df.index, dtype=int)
        return ones, 1

    watr = calculate_atr(wdf, atr_len).values
    whl2 = ((wdf["High"] + wdf["Low"]) / 2.0).values
    wclose = wdf["Close"].values
    wopen = wdf["Open"].values

    n_w = len(wdf)
    w_reg = np.zeros(n_w, dtype=int)
    trail_up, trail_dn = np.nan, np.nan
    reg = 0
    warmup_w = min(max(atr_len, 10), n_w - 1)

    for i in range(n_w):
        if i < warmup_w:
            w_reg[i] = 0
            continue
        a = watr[i] if not np.isnan(watr[i]) else 0.0
        ub = whl2[i] - factor * a
        db = whl2[i] + factor * a
        if np.isnan(trail_up):
            trail_up = ub
        if np.isnan(trail_dn):
            trail_dn = db
        if reg == 0:
            reg = 1 if wclose[i] > wopen[i] else -1
            trail_up, trail_dn = ub, db
        if reg == 1:
            trail_up = max(ub, trail_up)
            if wclose[i] < trail_up:
                reg = -1
                trail_dn = db
        else:
            trail_dn = min(db, trail_dn)
            if wclose[i] > trail_dn:
                reg = 1
                trail_up = ub
        w_reg[i] = reg

    w_series = pd.Series(w_reg, index=wdf.index)
    causal_daily = w_series.shift(1).reindex(df.index, method="ffill").fillna(0).astype(int)
    live_weekly = int(w_reg[-1]) if w_reg[-1] != 0 else 1
    return causal_daily, live_weekly


def calculate_position_size(
    portfolio_eur: float,
    risk_pct: float,
    max_pos_pct: float,
    entry_price: float,
    stop_price: float,
    is_crypto: bool = False,
) -> Dict[str, object]:
    """Calculates ATR/Stop-based position sizing."""
    if entry_price <= 0 or portfolio_eur <= 0 or risk_pct <= 0:
        return {"valid": False, "reason": "Ungültige Eingaben"}

    stop_dist_pct = ((entry_price - stop_price) / entry_price) * 100.0
    if stop_dist_pct <= 0.5:
        return {
            "valid": False,
            "stop_dist_pct": stop_dist_pct,
            "reason": "Kein aktiver Long-Stop unter aktuellem Kurs",
        }

    eff_stop_frac = max(stop_dist_pct / 100.0, 0.03)
    risk_budget_eur = portfolio_eur * (risk_pct / 100.0)
    max_alloc_eur = portfolio_eur * (max_pos_pct / 100.0)

    raw_alloc_eur = risk_budget_eur / eff_stop_frac
    alloc_eur = min(raw_alloc_eur, max_alloc_eur)
    capped = raw_alloc_eur > max_alloc_eur

    if is_crypto:
        shares = round(alloc_eur / entry_price, 4)
    else:
        shares = int(alloc_eur // entry_price)
        if shares > 0:
            alloc_eur = shares * entry_price

    actual_risk_eur = alloc_eur * (stop_dist_pct / 100.0)
    actual_risk_pct = (actual_risk_eur / portfolio_eur) * 100.0

    return {
        "valid": True,
        "shares": shares,
        "alloc_eur": alloc_eur,
        "alloc_pct": (alloc_eur / portfolio_eur) * 100.0,
        "stop_dist_pct": stop_dist_pct,
        "actual_risk_eur": actual_risk_eur,
        "actual_risk_pct": actual_risk_pct,
        "capped": capped,
        "is_penny": entry_price < 2.0,
    }


def _align_macro_flag(df_index: pd.DatetimeIndex, macro_df: Optional[pd.DataFrame], mode: str) -> np.ndarray:
    n = len(df_index)
    if mode == "off" or macro_df is None or macro_df.empty:
        return np.ones(n, dtype=int)
    col_map = {
        "fed_plus_qqq50": "macro_fed_plus_qqq50",
        "fed_pause": "macro_fed_pause",
        "strict_liq": "macro_strict_liq",
    }
    col = col_map.get(mode, "macro_fed_plus_qqq50")
    if col not in macro_df.columns:
        col = "macro_fed_pause"
    if col not in macro_df.columns:
        return np.ones(n, dtype=int)
    aligned = macro_df[col].reindex(df_index, method="ffill").fillna(1).astype(int)
    return aligned.values


def run_strategy(
    df_in: pd.DataFrame,
    ticker: str = "",
    comp_name: str = "",
    config: Optional[Dict[str, object]] = None,
    macro_df: Optional[pd.DataFrame] = None,
) -> Optional[StrategyResult]:
    """Runs the Early-Cycle + Macro-Liquidity Supertrend strategy."""
    if df_in is None or len(df_in) < 60:
        return None

    cfg = {**DEFAULT_STRATEGY_CONFIG, **(config or {})}
    atr_len = int(cfg["atr_len"])
    base_factor = float(cfg["base_factor"])
    macro_mode = str(cfg.get("macro_filter_mode", "fed_pause"))
    max_ext50 = float(cfg.get("anti_chase_max_ext_pct", 25.0))
    fresh_bars = max(1, int(cfg.get("fresh_flip_bars", 3)))
    use_weekly = bool(cfg.get("use_weekly_filter", False))
    use_div = bool(cfg.get("use_div_adaptive", False))
    div_sens = float(cfg["div_sens"]) if use_div else 0.0
    piv_len = int(cfg["piv_len"])
    osc_len = int(cfg["osc_len"])
    osc_smooth = int(cfg["osc_smooth"])
    div_bars = int(cfg["div_bars"])
    exec_next_open = bool(cfg["exec_next_open"])
    fee = float(cfg["fee_pct"]) / 100.0
    slip = float(cfg["slippage_pct"]) / 100.0

    df = df_in.copy().sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df["ATR"] = calculate_atr(df, atr_len)
    df["HL2"] = (df["High"] + df["Low"]) / 2.0
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = calculate_rsi(df["Close"], 14)

    if use_div:
        df["Osc"] = calculate_osc(df, osc_len, osc_smooth)
        osc = df["Osc"].values
    else:
        osc = np.zeros(len(df), dtype=float)

    weekly_causal, live_weekly_reg = calculate_weekly_supertrend_regime(df, atr_len, base_factor)
    df["weekly_regime"] = weekly_causal.values

    macro_ok_arr = _align_macro_flag(df.index, macro_df, macro_mode)
    df["macro_ok"] = macro_ok_arr

    atr = df["ATR"].values
    hl2 = df["HL2"].values
    sma50 = df["SMA50"].values
    close = df["Close"].values
    low = df["Low"].values
    high = df["High"].values
    op = df["Open"].values
    w_arr = df["weekly_regime"].values
    dates = df.index
    n = len(df)

    # 63-day (~3M) volatility-adjusted Momentum
    ret_63 = df["Close"].pct_change(63)
    vol_63 = df["Close"].pct_change().rolling(63).std() * np.sqrt(252.0)
    mom_series = (ret_63 / vol_63.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    latest_mom = float(mom_series.iloc[-1]) if pd.notna(mom_series.iloc[-1]) else 0.0

    regime = 0
    trail_up, trail_dn = np.nan, np.nan
    bars_since_div, active_div_type = 9999, 0
    last_pl_osc, last_pl_low = np.nan, np.nan
    last_ph_osc, last_ph_high = np.nan, np.nan

    equity = 100.0
    position_active = False
    held_units = 0.0
    entry_price = 0.0
    entry_equity = 100.0
    entry_date = None
    pending_action = 0

    regimes_out = np.zeros(n, dtype=int)
    stop_line_out = np.full(n, np.nan, dtype=float)
    equity_out = np.empty(n, dtype=float)
    buy_markers = np.full(n, np.nan, dtype=float)
    sell_markers = np.full(n, np.nan, dtype=float)
    trades: List[Dict[str, object]] = []

    warmup = max(atr_len, osc_len + osc_smooth, piv_len + 2, 50)
    bars_since_buy_flip = 9999
    last_daily_flip_idx = 0
    last_signal_change_idx = 0
    prev_state = "SELL"

    for i in range(n):
        if exec_next_open and pending_action != 0:
            exec_px = op[i]
            if pending_action == 1 and not position_active and exec_px > 0:
                bp = exec_px * (1.0 + slip)
                entry_price = bp
                entry_equity = equity
                entry_date = dates[i]
                equity *= (1.0 - fee)
                held_units = equity / bp
                position_active = True
                buy_markers[i] = bp
            elif pending_action == -1 and position_active and exec_px > 0:
                sp = exec_px * (1.0 - slip)
                new_eq = held_units * sp * (1.0 - fee)
                gain_pct = ((new_eq / entry_equity) - 1.0) * 100.0
                days_held = int((dates[i] - entry_date).days) if entry_date is not None else 0
                trades.append(
                    {
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": dates[i],
                        "exit_price": sp,
                        "return_pct": gain_pct,
                        "days_held": days_held,
                    }
                )
                equity = new_eq
                held_units = 0.0
                position_active = False
                sell_markers[i] = sp
            pending_action = 0

        if i < warmup:
            equity_out[i] = equity
            continue

        price = close[i]

        if use_div:
            p_idx = i - piv_len
            is_pl, is_ph = False, False
            if p_idx - piv_len >= 0 and p_idx + piv_len < n:
                osc_slice = osc[p_idx - piv_len : p_idx + piv_len + 1]
                if not np.isnan(osc[p_idx]):
                    if osc[p_idx] == np.min(osc_slice):
                        is_pl = True
                    if osc[p_idx] == np.max(osc_slice):
                        is_ph = True

            bull, bear = False, False
            if is_pl:
                if not np.isnan(last_pl_osc) and low[p_idx] < last_pl_low and osc[p_idx] > last_pl_osc:
                    bull = True
                last_pl_osc, last_pl_low = osc[p_idx], low[p_idx]
            if is_ph:
                if not np.isnan(last_ph_osc) and high[p_idx] > last_ph_high and osc[p_idx] < last_ph_osc:
                    bear = True
                last_ph_osc, last_ph_high = osc[p_idx], high[p_idx]

            if bull:
                bars_since_div, active_div_type = 0, 1
            elif bear:
                bars_since_div, active_div_type = 0, -1
            else:
                bars_since_div += 1

            factor = (
                base_factor * (1.0 - div_sens)
                if bars_since_div <= div_bars
                and ((regime == 1 and active_div_type == -1) or (regime == -1 and active_div_type == 1))
                else base_factor
            )
        else:
            factor = base_factor

        curr_atr = atr[i] if not np.isnan(atr[i]) else 0.0
        up_band = hl2[i] - factor * curr_atr
        dn_band = hl2[i] + factor * curr_atr

        if np.isnan(trail_up):
            trail_up = up_band
        if np.isnan(trail_dn):
            trail_dn = dn_band

        prev_regime = regime
        if regime == 0:
            regime = 1 if price > op[i] else -1
            trail_up, trail_dn = up_band, dn_band

        if regime == 1:
            trail_up = max(up_band, trail_up)
            if price < trail_up:
                regime = -1
                trail_dn = dn_band
                bars_since_div = 9999
                bars_since_buy_flip = 9999
            else:
                if prev_regime != 1:
                    bars_since_buy_flip = 0
                else:
                    bars_since_buy_flip += 1
        else:
            trail_dn = min(dn_band, trail_dn)
            if price > trail_dn:
                regime = 1
                trail_up = up_band
                bars_since_div = 9999
                bars_since_buy_flip = 0
            else:
                bars_since_buy_flip = 9999

        if regime != prev_regime:
            last_daily_flip_idx = i

        active_line = trail_up if regime == 1 else trail_dn
        regimes_out[i] = regime
        stop_line_out[i] = active_line

        ext50_now = ((price / sma50[i]) - 1.0) * 100.0 if (not np.isnan(sma50[i]) and sma50[i] > 0) else 0.0
        not_extended = (max_ext50 <= 0.0) or (ext50_now <= max_ext50)
        is_fresh = bars_since_buy_flip < fresh_bars
        weekly_ok = (w_arr[i] == 1) if use_weekly else True
        m_ok = macro_ok_arr[i] == 1

        # Entry requires: Daily UP + Fresh Flip + Anti-Chase + Macro Liquidity + optional Weekly
        allow_buy = (regime == 1) and is_fresh and not_extended and m_ok and weekly_ok

        # Logical signal state for display & signal_date tracking
        if regime == 1:
            if position_active or allow_buy:
                curr_state = "BUY"
            elif not not_extended:
                curr_state = "EXTENDED"
            elif not m_ok or not weekly_ok:
                curr_state = "WAIT"
            else:
                curr_state = "BUY" if (not_extended and m_ok and weekly_ok) else "WAIT"
        else:
            curr_state = "SELL"

        if curr_state != prev_state:
            last_signal_change_idx = i
            prev_state = curr_state

        if not exec_next_open:
            if allow_buy and not position_active:
                bp = price * (1.0 + slip)
                entry_price = bp
                entry_equity = equity
                entry_date = dates[i]
                equity *= (1.0 - fee)
                held_units = equity / bp
                position_active = True
                buy_markers[i] = bp
            elif regime == -1 and position_active:
                sp = price * (1.0 - slip)
                new_eq = held_units * sp * (1.0 - fee)
                gain_pct = ((new_eq / entry_equity) - 1.0) * 100.0
                days_held = int((dates[i] - entry_date).days) if entry_date is not None else 0
                trades.append(
                    {
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": dates[i],
                        "exit_price": sp,
                        "return_pct": gain_pct,
                        "days_held": days_held,
                    }
                )
                equity = new_eq
                held_units = 0.0
                position_active = False
                sell_markers[i] = sp
        else:
            if allow_buy and not position_active:
                pending_action = 1
            elif regime == -1 and position_active:
                pending_action = -1

        equity_out[i] = held_units * price if position_active else equity

    df["regime"] = regimes_out
    df["stop_line"] = stop_line_out
    df["buy_marker"] = buy_markers
    df["sell_marker"] = sell_markers

    eq_series = pd.Series(equity_out, index=dates, name="Strategy")
    bh_series = (df["Close"] / df["Close"].iloc[0]) * 100.0
    bh_series.name = "BuyHold"

    final_daily_reg = int(regimes_out[-1]) if regimes_out[-1] != 0 else -1
    final_weekly_reg = live_weekly_reg
    final_macro_ok = int(macro_ok_arr[-1])
    last_close = float(close[-1])
    last_sma50 = float(sma50[-1]) if not np.isnan(sma50[-1]) else last_close
    ext50_pct = ((last_close / last_sma50) - 1.0) * 100.0 if last_sma50 > 0 else 0.0
    not_ext_now = (max_ext50 <= 0.0) or (ext50_pct <= max_ext50)

    if final_daily_reg == 1:
        if position_active or pending_action == 1:
            final_sig = "BUY"
        elif not not_ext_now:
            final_sig = "EXTENDED"
        elif final_macro_ok == 0 or (use_weekly and final_weekly_reg != 1):
            final_sig = "WAIT"
        else:
            final_sig = "BUY"
    else:
        final_sig = "SELL"

    sig_idx = last_signal_change_idx if last_signal_change_idx > 0 else last_daily_flip_idx
    sig_date_str = dates[sig_idx].strftime("%Y-%m-%d")

    last_stop = float(stop_line_out[-1]) if not np.isnan(stop_line_out[-1]) else last_close
    stop_dist_pct = ((last_close - last_stop) / last_close * 100.0) if last_close > 0 else 0.0
    last_atr = float(atr[-1]) if not np.isnan(atr[-1]) else 0.0
    atr_pct = (last_atr / last_close * 100.0) if last_close > 0 else 0.0
    last_rsi = float(df["RSI"].iloc[-1]) if pd.notna(df["RSI"].iloc[-1]) else 50.0

    s_ret = float((eq_series.iloc[-1] / eq_series.iloc[0] - 1.0) * 100.0)
    b_ret = float((bh_series.iloc[-1] / bh_series.iloc[0] - 1.0) * 100.0)
    s_dd = calculate_max_drawdown(eq_series)
    b_dd = calculate_max_drawdown(bh_series)

    years = max((dates[-1] - dates[0]).days / 365.25, 0.1)
    strat_cagr = float(((eq_series.iloc[-1] / eq_series.iloc[0]) ** (1.0 / years) - 1.0) * 100.0)
    dret = eq_series.pct_change().dropna()
    sharpe = float((dret.mean() / dret.std()) * np.sqrt(252.0)) if dret.std() > 0 else 0.0
    calmar = float(strat_cagr / abs(s_dd)) if s_dd < -1.0 else 0.0

    wins = [float(t["return_pct"]) for t in trades if float(t["return_pct"]) > 0]
    losses = [float(t["return_pct"]) for t in trades if float(t["return_pct"]) <= 0]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    sum_wins = sum(wins)
    sum_losses = abs(sum(losses))
    if sum_losses > 0:
        pf = min(sum_wins / sum_losses, 99.9)
    elif sum_wins > 0:
        pf = 99.9
    else:
        pf = 0.0

    report_text = build_report_text(
        ticker=ticker,
        comp_name=comp_name or ticker,
        cfg=cfg,
        final_sig=final_sig,
        sig_date=sig_date_str,
        daily_reg=final_daily_reg,
        weekly_reg=final_weekly_reg,
        macro_ok=final_macro_ok,
        last_close=last_close,
        last_stop=last_stop,
        stop_dist_pct=stop_dist_pct,
        ext50_pct=ext50_pct,
        atr_pct=atr_pct,
        mom_score=latest_mom,
        rsi_14=last_rsi,
        eq_series=eq_series,
        bh_series=bh_series,
        s_ret=s_ret,
        b_ret=b_ret,
        s_cagr=strat_cagr,
        s_dd=s_dd,
        b_dd=b_dd,
        sharpe=sharpe,
        calmar=calmar,
        pf=pf,
        trades=trades,
        wins=wins,
        losses=losses,
    )

    return StrategyResult(
        equity_curve=eq_series,
        benchmark_curve=bh_series,
        df_indicators=df,
        trades=trades,
        current_signal=final_sig,
        daily_regime=final_daily_reg,
        weekly_regime=final_weekly_reg,
        macro_ok=final_macro_ok,
        signal_date=sig_date_str,
        last_close=last_close,
        stop_price=last_stop,
        stop_dist_pct=stop_dist_pct,
        ext50_pct=ext50_pct,
        mom_score=latest_mom,
        atr_pct=atr_pct,
        rsi_14=last_rsi,
        strat_return=s_ret,
        bh_return=b_ret,
        strat_cagr=strat_cagr,
        strat_dd=s_dd,
        bh_dd=b_dd,
        sharpe=sharpe,
        calmar=calmar,
        profit_factor=pf,
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        num_trades=len(trades),
        report_text=report_text,
    )


def build_report_text(
    ticker: str,
    comp_name: str,
    cfg: Dict[str, object],
    final_sig: str,
    sig_date: str,
    daily_reg: int,
    weekly_reg: int,
    macro_ok: int,
    last_close: float,
    last_stop: float,
    stop_dist_pct: float,
    ext50_pct: float,
    atr_pct: float,
    mom_score: float,
    rsi_14: float,
    eq_series: pd.Series,
    bh_series: pd.Series,
    s_ret: float,
    b_ret: float,
    s_cagr: float,
    s_dd: float,
    b_dd: float,
    sharpe: float,
    calmar: float,
    pf: float,
    trades: List[Dict[str, object]],
    wins: List[float],
    losses: List[float],
) -> str:
    w_mean, w_med, w_q1, w_q3 = get_detailed_stats(wins)
    l_mean, l_med, l_q1, l_q3 = get_detailed_stats(losses)
    wr = (len(wins) / len(trades) * 100.0) if trades else 0.0

    mode_str = "Klassisch ST" if not cfg["use_div_adaptive"] else "Divergenz-ST"
    macro_labels = {
        "fed_plus_qqq50": "Makro: Rezessionsfest (Fed+QQQ>50, 3T)",
        "fed_pause": "Makro: Fed-Pause/Cut (3T)",
        "strict_liq": "Makro: Streng (Fed+DXY/10Y, 3T)",
        "off": "Makro: Aus",
    }
    m_str = macro_labels.get(str(cfg.get("macro_filter_mode", "fed_pause")), "Makro")
    ext_cap = float(cfg.get("anti_chase_max_ext_pct", 25.0))
    ac_str = f"Anti-Chase <+{ext_cap:.0f}% SMA50" if ext_cap > 0 else "ohne Anti-Chase"
    exec_str = "Open[i+1]" if cfg["exec_next_open"] else "Close[i]"

    d_lbl = "BULL (UP)" if daily_reg == 1 else "BEAR (DOWN)"
    m_lbl = "GRÜN (ON)" if macro_ok == 1 else "ROT (OFF)"
    w_lbl = "BULL" if weekly_reg == 1 else "BEAR"

    is_crypto = "-USD" in ticker.upper()
    pos_info = calculate_position_size(
        portfolio_eur=float(cfg["portfolio_eur"]),
        risk_pct=float(cfg["risk_per_trade_pct"]),
        max_pos_pct=float(cfg["max_position_pct"]),
        entry_price=last_close,
        stop_price=last_stop,
        is_crypto=is_crypto,
    )

    lines = [
        f"ANALYSE-BERICHT: {comp_name} ({ticker})",
        f"Setup: {mode_str} ({cfg['atr_len']}/{cfg['base_factor']}) | {m_str} | {ac_str} | {exec_str}",
        "=" * 72,
        f"STATUS:   {final_sig:<8} (seit {sig_date}) | Daily: {d_lbl} | Makro: {m_lbl} | Weekly: {w_lbl}",
        f"CHART:    Kurs {last_close:.2f} | Stop {last_stop:.2f} ({stop_dist_pct:+.1f}%) | vs SMA50: {ext50_pct:+.1f}% | ATR: {atr_pct:.1f}%",
        f"MOMENTUM: 3M Vol-Adj {mom_score:+.2f} | RSI(14): {rsi_14:.1f}",
    ]

    if pos_info.get("valid") and final_sig == "BUY":
        cap_note = " [Max-Pos gekappt]" if pos_info["capped"] else ""
        penny_note = " [PENNYSTOCK <$2!]" if pos_info["is_penny"] else ""
        lines.append(
            f"POS-GRÖSSE ({float(cfg['portfolio_eur']):,.0f}€ / {cfg['risk_per_trade_pct']}% Risk): "
            f"{pos_info['shares']} Stk ≈ {pos_info['alloc_eur']:,.0f}€ ({pos_info['alloc_pct']:.1f}%) "
            f"| Risiko: {pos_info['actual_risk_eur']:,.0f}€{cap_note}{penny_note}"
        )
    elif final_sig == "WAIT":
        lines.append("START-RAMPE: Daily ist sauber UP & nicht überdehnt — wartet auf Makro-Liquidität GRÜN!")
    elif final_sig == "EXTENDED":
        lines.append(f"ANTI-CHASE: Kurs bereits {ext50_pct:+.1f}% über SMA50 — Rücksetzer abwarten statt hinterherlaufen!")
    else:
        lines.append("POS-GRÖSSE: Kein Long-Signal aktiv (Cash / Stop ausgelöst).")

    diff = s_ret - b_ret
    verdict = "OK (vs B&H)" if diff > 0 else "Hinter B&H"
    lines.extend(
        [
            "-" * 72,
            f"RENDITE:  Strategie {s_ret:+.1f}% (CAGR {s_cagr:+.1f}%) vs B&H {b_ret:+.1f}% -> {verdict}",
            f"RISIKO:   MaxDD Strat {s_dd:.1f}% vs B&H {b_dd:.1f}% | Sharpe {sharpe:.2f} | Calmar {calmar:.2f}",
            f"TRADES:   {len(trades)} ({len(wins)} W / {len(losses)} L) | WinRate {wr:.1f}% | Profit Factor {pf:.2f}",
            f"  GEWINNE:  Ø {w_mean:+.1f}% | Med {w_med:+.1f}% | Q1 {w_q1:+.1f}% / Q3 {w_q3:+.1f}%",
            f"  VERLUSTE: Ø {l_mean:+.1f}% | Med {l_med:+.1f}% | Q1 {l_q1:+.1f}% / Q3 {l_q3:+.1f}%",
            "-" * 72,
            "JAHRES-VERGLEICH:",
        ]
    )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])

    df_c = pd.DataFrame({"S": eq_series, "B": bh_series})
    for year, grp in df_c.groupby(df_c.index.year):
        yr_s = (grp["S"].iloc[-1] / grp["S"].iloc[0] - 1.0) * 100.0
        yr_b = (grp["B"].iloc[-1] / grp["B"].iloc[0] - 1.0) * 100.0
        t_cnt, w_cnt, l_cnt = 0, 0, 0
        if not trades_df.empty:
            yt = trades_df[trades_df["exit_date"].dt.year == year]
            t_cnt = len(yt)
            w_cnt = int((yt["return_pct"] > 0).sum())
            l_cnt = t_cnt - w_cnt
        lines.append(
            f"  {year:<4} | Strat {yr_s:>+7.1f}% (B&H {yr_b:>+7.1f}%) | Trades: {t_cnt:>2} ({w_cnt}W/{l_cnt}L)"
        )

    if trades:
        lines.append("-" * 72)
        lines.append("LETZTE 5 ABGESCHLOSSENE TRADES:")
        for tr in trades[-5:][::-1]:
            ed = tr["entry_date"].strftime("%Y-%m-%d") if tr["entry_date"] is not None else "?"
            xd = tr["exit_date"].strftime("%Y-%m-%d") if tr["exit_date"] is not None else "?"
            lines.append(
                f"  {ed} -> {xd} ({tr['days_held']:>3}d) | "
                f"{tr['entry_price']:>7.2f} -> {tr['exit_price']:>7.2f} | {tr['return_pct']:>+6.1f}%"
            )

    return "\n".join(lines)
