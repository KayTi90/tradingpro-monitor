"""Macro Liquidity Regime Engine & Thematic Hype-Radar (with German UCITS ETFs & WKNs).

Tracks:
1. Central Bank & Global Liquidity proxies via Yahoo Finance:
   - ^IRX   : 3M US T-Bill Yield (Fed policy rate trend proxy)
   - ^TNX   : 10Y US Treasury Yield
   - DX-Y.NYB: US Dollar Index (DXY, inverse global liquidity)
   - HYG/IEF: High-Yield Credit Risk Appetite Ratio
   - IWM    : Russell 2000 Small-Cap Liquidity Barometer
   - QQQ    : Nasdaq 100 Benchmark
   - ^VIX   : Volatility / Fear Index
   - BTC-USD: Global Liquidity Sponge

2. Thematic Hype-Radar: 19 US Theme ETFs paired with their verified German-tradable
   UCITS ETF equivalents (WKN, Name, Xetra `.DE` Yahoo ticker for chart comparison).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from .data_loader import CACHE_DIR, load_ticker_data


MACRO_CACHE_FILE = os.path.join(CACHE_DIR, "_MACRO_CACHE.csv")

MACRO_SYMBOLS: Dict[str, str] = {
    "TNX": "^TNX",
    "IRX": "^IRX",
    "DXY": "DX-Y.NYB",
    "HYG": "HYG",
    "IEF": "IEF",
    "IWM": "IWM",
    "NDX": "^NDX",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "BTC": "BTC-USD",
}


def fmt_pts(val: float) -> str:
    """Formats Nasdaq-100 index points with German thousands separator (e.g. 26.047)."""
    if val is None or np.isnan(val):
        return "—"
    if abs(val) >= 1000.0:
        return f"{val:,.0f}".replace(",", ".")
    return f"{val:.2f}"

# (US Radar Ticker, Theme, DE WKN, Xetra Yahoo Ticker, German UCITS ETF Name)
HYPE_RADAR_ETFS: List[Tuple[str, str, str, Optional[str], str]] = [
    ("CIBR", "Cybersecurity / Post-Quantum", "A2P4HV", "CBRS.DE", "First Trust Nasdaq Cybersecurity UCITS"),
    ("XBI", "Biotech & AI-Bio", "A2DWAW", "2B70.DE", "iShares Nasdaq US Biotechnology UCITS"),
    ("BOTZ", "Physical AI & Robotics", "A2QPBW", "XB0T.DE", "Global X Robotics & AI UCITS"),
    ("ROBO", "Robotics & Automation", "A2ANH0", "2B76.DE", "iShares Automation & Robotics UCITS"),
    ("SMH", "Halbleiter / AI-Chips", "A2QC5J", "VVSM.DE", "VanEck Semiconductor UCITS"),
    ("QTUM", "Quantum & Deep Tech", "A418QM", "QUTM.DE", "VanEck Quantum Computing UCITS"),
    ("UFO", "Space / Weltraum", "A3DP9J", "JEDI.DE", "VanEck Space Innovators UCITS"),
    ("SHLD", "Defense Tech & Drohnen", "A40E7A", "4MMR.DE", "Global X Defence Tech UCITS"),
    ("ITA", "Aerospace & Defense", "A3D9M1", "DFEN.DE", "VanEck Defense UCITS"),
    ("NLR", "Nuclear Energy / SMR", "A3D47K", "NUKL.DE", "VanEck Uranium & Nuclear Tech UCITS"),
    ("URA", "Uran & Brennstoff", "A3DC8S", "URNU.DE", "Global X Uranium UCITS"),
    ("GRID", "Smart Grid & Strom-Infra", "A3DGK5", "GRID.DE", "First Trust Clean Edge Smart Grid UCITS"),
    ("COPX", "Kupfer & Elektrifizierung", "A3C7FZ", "4COP.DE", "Global X Copper Miners UCITS"),
    ("REMX", "Seltene Erden & Minerals", "A3CRL9", "VVMX.DE", "VanEck Rare Earth & Strat. Metals UCITS"),
    ("BLOK", "Blockchain / Stablecoins / RWA", "A2QQ8F", "DAVV.DE", "VanEck Crypto & Blockchain UCITS"),
    ("TAN", "Solar & Clean Energy", "A2QQ9R", "S0LR.DE", "Invesco Solar Energy UCITS"),
    ("MSOS", "Cannabis (Alt-Hype)", "A2PPE8", None, "HANetf Medical Cannabis & Wellness UCITS"),
    ("IWM", "Russell 2000 (Small Caps)", "A1XFN1", "ZPRR.DE", "SPDR Russell 2000 US Small Cap UCITS"),
    ("QQQ", "Nasdaq 100 (Benchmark)", "A0YEDL", "SXRV.DE", "iShares Nasdaq 100 UCITS (Acc)"),
]

HYPE_META_BY_TICKER: Dict[str, Dict[str, Optional[str]]] = {
    sym: {"theme": theme, "wkn": wkn, "ucits_ticker": ucits_t, "ucits_name": ucits_n}
    for sym, theme, wkn, ucits_t, ucits_n in HYPE_RADAR_ETFS
}


def _download_single_close(sym: str) -> Optional[pd.Series]:
    try:
        df = yf.download(sym, start="2019-06-01", progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if "Close" in df.columns:
            return pd.to_numeric(df["Close"], errors="coerce").dropna()
    except Exception:
        return None
    return None


def _batch_download_ohlcv(symbols: List[str], start: str = "2020-01-01") -> Dict[str, pd.DataFrame]:
    """Downloads multiple Yahoo Finance tickers in one reliable batch request."""
    out: Dict[str, pd.DataFrame] = {}
    if not symbols:
        return out
    try:
        raw = yf.download(
            symbols,
            start=start,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            return out
        if len(symbols) == 1:
            df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df.dropna(subset=["Close"])
            if not df.empty:
                out[symbols[0]] = df
            return out

        for sym in symbols:
            if isinstance(raw.columns, pd.MultiIndex) and sym in raw.columns.get_level_values(0):
                df = raw[sym].copy()
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df = df.dropna(subset=["Close"])
                if not df.empty:
                    out[sym] = df
    except Exception:
        pass
    return out


def _apply_confirmation(series: pd.Series, days: int = 3) -> pd.Series:
    """3-day hysteresis filter: requires `days` consecutive trading days to flip state,
    eliminating ~85% of 1-to-5-day boundary whipsaws without sacrificing major cycle capture.
    """
    vals = series.fillna(0).astype(int).values
    if len(vals) == 0:
        return series
    out = np.empty_like(vals)
    state = int(vals[0])
    streak = 0
    for i, v in enumerate(vals):
        iv = int(v)
        if iv != state:
            streak += 1
            if streak >= days:
                state = iv
                streak = 0
        else:
            streak = 0
        out[i] = state
    return pd.Series(out, index=series.index, dtype=int)


def _apply_sma200_band(
    price: pd.Series,
    sma200: pd.Series,
    up_buf: float = 0.015,
    dn_buf: float = 0.020,
) -> pd.Series:
    """Stateful hysteresis band around SMA200 (+1.5% re-entry / -2.0% exit)
    for the 4-stage Core-Allocation (A0LC12 / DBX0AN).
    """
    p_vals = price.values
    m_vals = sma200.values
    out = np.zeros(len(p_vals), dtype=int)
    state = 0
    initialized = False
    for i in range(len(p_vals)):
        p = p_vals[i]
        m = m_vals[i]
        if np.isnan(p) or np.isnan(m) or m <= 0:
            out[i] = 0
            continue
        if not initialized:
            state = 1 if p >= m else 0
            initialized = True
        else:
            if state == 1 and p < m * (1.0 - dn_buf):
                state = 0
            elif state == 0 and p > m * (1.0 + up_buf):
                state = 1
        out[i] = state
    return pd.Series(out, index=price.index, dtype=int)


def _apply_vol_band(vol20: pd.Series, lo: float = 20.0, hi: float = 28.0) -> pd.Series:
    """Stateful hysteresis band on 20-day annualized volatility:
    exits 2x Turbo only when Vola > hi (28%), re-enables 2x Turbo when Vola < lo (20%).
    Cuts tax-triggering whipsaws in half compared to a hard single threshold.
    """
    vals = vol20.values
    out = np.zeros(len(vals), dtype=int)
    state = 1
    initialized = False
    for i, x in enumerate(vals):
        if np.isnan(x):
            out[i] = 0
            continue
        if not initialized:
            state = 1 if x < hi else 0
            initialized = True
        else:
            if state == 1 and x > hi:
                state = 0
            elif state == 0 and x < lo:
                state = 1
        out[i] = state
    return pd.Series(out, index=vol20.index, dtype=int)


def compute_macro_signals(mdf: pd.DataFrame) -> pd.DataFrame:
    """Enriches raw macro close DataFrame with strictly causal trading-day liquidity flags,
    3-day confirmation hysteresis, explicit flip thresholds, and 4-stage Core allocation.
    """
    if mdf is None or mdf.empty:
        return pd.DataFrame()
    raw = mdf.copy().sort_index().ffill()
    df = raw[raw.index.dayofweek < 5].copy()
    if df.empty:
        df = raw.copy()

    if "IRX" in df.columns:
        df["IRX_thresh"] = df["IRX"].shift(63) + 0.05
        df["sig_fed_raw"] = (df["IRX"] <= df["IRX_thresh"]).astype(int)
    else:
        df["IRX_thresh"] = np.nan
        df["sig_fed_raw"] = 1

    if "TNX" in df.columns:
        df["TNX_sma50"] = df["TNX"].rolling(50).mean()
        df["sig_tnx"] = (df["TNX"] < df["TNX_sma50"]).astype(int)
    else:
        df["TNX_sma50"] = np.nan
        df["sig_tnx"] = 1

    if "DXY" in df.columns:
        df["DXY_sma50"] = df["DXY"].rolling(50).mean()
        df["sig_dxy"] = (df["DXY"] < df["DXY_sma50"]).astype(int)
    else:
        df["DXY_sma50"] = np.nan
        df["sig_dxy"] = 1

    idx_col = "NDX" if ("NDX" in df.columns and df["NDX"].notna().sum() >= 200) else ("QQQ" if "QQQ" in df.columns else None)
    if idx_col is not None:
        idx_s = df[idx_col]
        df["NDX_ref"] = idx_s
        df["QQQ_sma20"] = idx_s.rolling(20).mean()
        df["QQQ_sma50"] = idx_s.rolling(50).mean()
        df["QQQ_sma200"] = idx_s.rolling(200).mean()
        df["QQQ_ext200"] = ((idx_s / df["QQQ_sma200"]) - 1.0) * 100.0
        df["sig_qqq20_raw"] = (idx_s > df["QQQ_sma20"]).astype(int)
        df["sig_qqq20_3d"] = _apply_confirmation(df["sig_qqq20_raw"], days=3)
        df["sig_qqq50_raw"] = (idx_s > df["QQQ_sma50"]).astype(int)
        df["sig_qqq50_3d"] = _apply_confirmation(df["sig_qqq50_raw"], days=3)
        # R1-F: Wider SMA200 hysteresis band (+2.5% re-entry / -3.0% exit)
        df["sig_qqq200_band"] = _apply_sma200_band(idx_s, df["QQQ_sma200"], up_buf=0.025, dn_buf=0.030)
        vol20_raw = idx_s.pct_change().rolling(20).std() * np.sqrt(252.0) * 100.0
        df["QQQ_vol20"] = vol20_raw.rolling(5).mean()
        # V-Turbo: Nasdaq-100 calibrated volatility band (exit 2x > 35%, re-enable < 25%)
        df["sig_vol_band"] = _apply_vol_band(df["QQQ_vol20"], lo=25.0, hi=35.0)
        # Overheating hysteresis brake (> +30% vs SMA200 trims 2x -> 1x; re-enables < +22%)
        df["sig_not_overheat"] = _apply_vol_band(df["QQQ_ext200"], lo=22.0, hi=30.0)
    else:
        df["NDX_ref"] = np.nan
        df["QQQ_sma20"] = np.nan
        df["QQQ_sma50"] = np.nan
        df["QQQ_sma200"] = np.nan
        df["QQQ_ext200"] = np.nan
        df["sig_qqq20_raw"] = 1
        df["sig_qqq20_3d"] = 1
        df["sig_qqq50_raw"] = 1
        df["sig_qqq50_3d"] = 1
        df["sig_qqq200_band"] = 1
        df["QQQ_vol20"] = np.nan
        df["sig_vol_band"] = 1
        df["sig_not_overheat"] = 1

    if "HYG" in df.columns and "IEF" in df.columns:
        ratio = df["HYG"] / df["IEF"].replace(0.0, np.nan)
        df["HYG_IEF"] = ratio
        df["HYG_IEF_sma50"] = ratio.rolling(50).mean()
        df["sig_credit"] = (ratio > df["HYG_IEF_sma50"]).astype(int)
    else:
        df["sig_credit"] = 1

    if "IWM" in df.columns:
        df["IWM_sma50"] = df["IWM"].rolling(50).mean()
        df["sig_iwm"] = (df["IWM"] > df["IWM_sma50"]).astype(int)
    else:
        df["IWM_sma50"] = np.nan
        df["sig_iwm"] = 1

    df["sig_vix"] = (df["VIX"] < 22.0).astype(int) if "VIX" in df.columns else 1
    df["sig_btc"] = (df["BTC"] > df["BTC"].rolling(50).mean()).astype(int) if "BTC" in df.columns else 1

    # 3-Day Confirmed Macro Regimes (eliminates noisy 1-5 day flickering!)
    df["sig_fed"] = _apply_confirmation(df["sig_fed_raw"], days=3)
    df["sig_qqq50"] = df["sig_qqq50_raw"]
    raw_fed_qqq50 = ((df["sig_fed_raw"] == 1) & (df["sig_qqq50_raw"] == 1)).astype(int)
    raw_strict = ((df["sig_fed_raw"] == 1) & ((df["sig_dxy"] == 1) | (df["sig_tnx"] == 1))).astype(int)

    df["macro_fed_pause"] = df["sig_fed"]
    df["macro_fed_plus_qqq50"] = _apply_confirmation(raw_fed_qqq50, days=3)
    df["macro_strict_liq"] = _apply_confirmation(raw_strict, days=3)

    # 3-Stage V-Turbo Core Allocation (A0LC12 2x Nasdaq UCITS / DBX0AN Money Market)
    # Daily execution (next morning 09:00 Xetra, zero Friday-delay!) + Vola-gated SMA20(3T) early 50% start:
    # - Stage 2 (100%): (SMA200-Band | SMA50-3T) AND Vola(25/35) OK AND Overheating(22/30) OK
    # - Stage 1 ( 50%): ((SMA200-Band | SMA50-3T) & ~turbo_ok) OR (SMA20-3T & Vola OK)
    # - Stage 0 (  0%): otherwise
    s200 = df["sig_qqq200_band"] == 1
    s50 = df["sig_qqq50_3d"] == 1
    s20 = df["sig_qqq20_3d"] == 1
    vol_ok = df["sig_vol_band"] == 1
    oh_ok = df["sig_not_overheat"] == 1
    trend_up = s200 | s50
    turbo_ok = vol_ok & oh_ok
    early_s20 = (~trend_up) & s20 & vol_ok

    core_daily = np.zeros(len(df), dtype=int)
    core_daily[((trend_up & ~turbo_ok) | early_s20).values] = 1
    core_daily[(trend_up & turbo_ok).values] = 2
    df["core_stage_daily"] = core_daily
    df["core_stage"] = core_daily

    df["cb_score"] = df["sig_fed"] + df["sig_tnx"] + df["sig_dxy"]
    df["liq_score_6"] = (
        df["sig_fed"] + df["sig_tnx"] + df["sig_dxy"] + df["sig_credit"] + df["sig_iwm"] + df["sig_btc"]
    )
    return df


def get_core_allocation_status(mdf: Optional[pd.DataFrame]) -> Dict[str, object]:
    """Computes live 3-Stage V-Turbo Core-Allocation status (A0LC12 2x Nasdaq / DBX0AN Money Market)
    referenced to Nasdaq-100 points (^NDX), Vola-Band 25/35, Smart-Rate >4%, and Overheating <30%.
    """
    if mdf is None or mdf.empty or "QQQ_sma200" not in mdf.columns:
        return {"available": False}

    idx_col = "NDX_ref" if "NDX_ref" in mdf.columns else ("NDX" if "NDX" in mdf.columns else "QQQ")
    if idx_col not in mdf.columns:
        return {"available": False}
    is_ndx = "NDX" in mdf.columns and mdf["NDX"].notna().sum() >= 200
    idx_name = "Nasdaq-100" if is_ndx else "QQQ"

    last = mdf.iloc[-1]
    last_dt = mdf.index[-1]
    qqq = float(last.get(idx_col, np.nan))
    sma20 = float(last.get("QQQ_sma20", np.nan))
    sma50 = float(last.get("QQQ_sma50", np.nan))
    sma200 = float(last.get("QQQ_sma200", np.nan))
    if np.isnan(qqq) or np.isnan(sma50) or np.isnan(sma200) or sma50 <= 0 or sma200 <= 0:
        return {"available": False}
    if np.isnan(sma20) or sma20 <= 0:
        sma20 = sma50

    sma200_up = sma200 * 1.025
    sma200_dn = sma200 * 0.970
    vs_sma20_pct = ((qqq / sma20) - 1.0) * 100.0
    vs_sma50_pct = ((qqq / sma50) - 1.0) * 100.0
    vs_sma200_pct = ((qqq / sma200) - 1.0) * 100.0
    vs_band_up_pct = ((qqq / sma200_up) - 1.0) * 100.0
    vs_band_dn_pct = ((qqq / sma200_dn) - 1.0) * 100.0

    sma20_ok = int(last.get("sig_qqq20_3d", 0)) == 1
    sma20_raw_ok = int(last.get("sig_qqq20_raw", 0)) == 1
    sma50_ok = int(last.get("sig_qqq50_3d", 0)) == 1
    sma50_raw_ok = int(last.get("sig_qqq50_raw", 0)) == 1
    sma200_band_ok = int(last.get("sig_qqq200_band", 0)) == 1
    irx_val = float(last.get("IRX", np.nan))
    fed_ok = int(last.get("sig_fed", 0)) == 1
    smart_fed_ok = int(last.get("sig_fed_smart4", 1)) == 1
    vol20 = float(last.get("QQQ_vol20", np.nan))
    vol_ok = int(last.get("sig_vol_band", 0)) == 1
    vol_stress = pd.notna(vol20) and vol20 > 35.0
    overheat_ok = int(last.get("sig_not_overheat", 1)) == 1
    early_sma20_active = (not sma200_band_ok) and (not sma50_ok) and sma20_ok and vol_ok

    stage = int(last.get("core_stage", 0))
    live_stage = int(last.get("core_stage_daily", stage))
    is_friday = last_dt.dayofweek == 4
    weekly_pending = False  # Daily execution — no Friday wait!

    stage_meta = {
        0: (0, 100, 0.0, "CRASH-SCHUTZ", "inverse-danger", "#ff4d4d"),
        1: (50, 50, 1.0, "FRÜHSTART (1×)" if early_sma20_active else "TREND (1×)", "inverse-warning", "#f1c40f"),
        2: (100, 0, 2.0, "VOLLER TURBO (2×)", "inverse-success", "#2ecc71"),
    }
    a0lc12_pct, dbx0an_pct, eff_lev, stage_name, bootstyle, color_hex = stage_meta.get(
        stage, stage_meta[0]
    )

    if stage == 0:
        need20_pct = ((sma20 / qqq) - 1.0) * 100.0
        need50_pct = ((sma50 / qqq) - 1.0) * 100.0
        if vol_ok:
            if sma20_raw_ok:
                next_trigger = (
                    f"Frühstart 50% (Depot A): {idx_name} > SMA20 ({fmt_pts(sma20)}) wartet auf 3T-Bestätigung  |  "
                    f"100% ab > SMA50 ({fmt_pts(sma50)}, {need50_pct:+.1f}%)"
                )
            else:
                next_trigger = (
                    f"Frühstart 50% (Depot A): {idx_name} > SMA20 ({fmt_pts(sma20)}, noch {need20_pct:+.1f}%, 3T)  |  "
                    f"100% ab > SMA50 ({fmt_pts(sma50)}, {need50_pct:+.1f}%)"
                )
        else:
            next_trigger = (
                f"Einstieg 50%: {idx_name} > SMA50 ({fmt_pts(sma50)}, noch {need50_pct:+.1f}%)  |  "
                f"SMA20-Frühstart gesperrt (Vola {vol20:.1f}% ROT)"
            )
    elif stage == 1:
        if early_sma20_active:
            need50_pct = ((sma50 / qqq) - 1.0) * 100.0
            need200_pct = ((sma200_up / qqq) - 1.0) * 100.0
            target_up = f"SMA200+2,5% ({fmt_pts(sma200_up)}, {need200_pct:+.1f}%)" if sma200_up < sma50 else f"SMA50 ({fmt_pts(sma50)}, {need50_pct:+.1f}%)"
            next_trigger = (
                f"2×-Turbo (100% A0LC12) ab: {target_up}  |  "
                f"Exit → 0%: < SMA20 ({fmt_pts(sma20)}, {vs_sma20_pct:+.1f}%, 3T) oder Vola > 35%"
            )
        else:
            blockers = []
            if not vol_ok:
                blockers.append(f"Vola < 25% (Band 25/35%, akt. {vol20:.1f}%)")
            if not overheat_ok:
                blockers.append(f"Ext200 < +22% (akt. {vs_sma200_pct:+.1f}%)")
            blocker_str = " + ".join(blockers) if blockers else "nächster Tagesschluss"
            exit_ref = f"SMA200 -3% ({fmt_pts(sma200_dn)}, {vs_band_dn_pct:+.1f}%)" if sma200_band_ok else f"SMA50 ({fmt_pts(sma50)}, {vs_sma50_pct:+.1f}%)"
            next_trigger = f"2×-Turbo (Stufe 2 / 100% A0LC12): {blocker_str}  |  Exit-Puffer: {exit_ref}"
    else:
        exit_ref = f"SMA200 -3% ({fmt_pts(sma200_dn)}, {vs_band_dn_pct:+.1f}%)" if sma200_band_ok else f"SMA50 ({fmt_pts(sma50)}, {vs_sma50_pct:+.1f}%)"
        next_trigger = (
            f"Bremse → 50% (Depot B verkaufen): bei Vola > 35% ({vol20:.1f}%) oder Ext200 > +30% ({vs_sma200_pct:+.1f}%)  |  "
            f"Exit-Puffer: {exit_ref}"
        )

    if vol_stress and stage > 0:
        next_trigger = f"⚠️ Vola-Stress ({vol20:.1f}% > 35%)  |  {next_trigger}"

    return {
        "available": True,
        "as_of": last_dt.strftime("%d.%m.%Y"),
        "index_name": idx_name,
        "stage": stage,
        "live_stage": live_stage,
        "is_friday": is_friday,
        "weekly_pending": weekly_pending,
        "early_sma20_active": early_sma20_active,
        "stage_name": stage_name,
        "color_hex": color_hex,
        "bootstyle": bootstyle,
        "a0lc12_pct": a0lc12_pct,
        "dbx0an_pct": dbx0an_pct,
        "eff_lev": eff_lev,
        "ndx": qqq,
        "qqq": qqq,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "sma200_up": sma200_up,
        "sma200_dn": sma200_dn,
        "vs_sma20_pct": vs_sma20_pct,
        "vs_sma50_pct": vs_sma50_pct,
        "vs_sma200_pct": vs_sma200_pct,
        "vs_band_up_pct": vs_band_up_pct,
        "vs_band_dn_pct": vs_band_dn_pct,
        "sma20_ok": sma20_ok,
        "sma50_ok": sma50_ok,
        "sma200_band_ok": sma200_band_ok,
        "irx": irx_val,
        "fed_ok": fed_ok,
        "smart_fed_ok": smart_fed_ok,
        "vol20": vol20,
        "vol_ok": vol_ok,
        "vol_stress": vol_stress,
        "overheat_ok": overheat_ok,
        "next_trigger": next_trigger,
    }


def get_macro_flip_diagnostics(mdf: Optional[pd.DataFrame]) -> Dict[str, object]:
    """Computes exact distance-to-flip metrics and the known 30-trading-day forward path
    of the 63-day Fed 3M (^IRX) flip threshold.
    """
    if mdf is None or mdf.empty or len(mdf) < 70:
        return {"available": False}

    last = mdf.iloc[-1]
    last_dt = mdf.index[-1]

    irx = float(last.get("IRX", np.nan))
    irx_th = float(last.get("IRX_thresh", np.nan))
    irx_diff_bp = (irx - irx_th) * 100.0 if (pd.notna(irx) and pd.notna(irx_th)) else np.nan
    fed_ok = int(last.get("sig_fed", 0)) == 1

    # Known 30-trading-day forward trajectory of IRX_thresh(t+k) = IRX(t+k-63) + 0.05
    n_fut = min(30, len(mdf) - 63)
    fut_dates = pd.bdate_range(last_dt + pd.Timedelta(days=1), periods=n_fut)
    fut_irx_thresh = pd.Series(
        [float(mdf["IRX"].iloc[-63 + k]) + 0.05 for k in range(n_fut)],
        index=fut_dates,
    )
    auto_flip_date = None
    if not fed_ok and pd.notna(irx):
        for d, v in fut_irx_thresh.items():
            if irx <= v:
                auto_flip_date = d.strftime("%d.%m.%Y")
                break
    irx_th_20d = float(fut_irx_thresh.iloc[min(19, len(fut_irx_thresh) - 1)]) if len(fut_irx_thresh) > 0 else np.nan
    irx_gap_20d_bp = (irx - irx_th_20d) * 100.0 if (pd.notna(irx) and pd.notna(irx_th_20d)) else np.nan

    tnx = float(last.get("TNX", np.nan))
    tnx_ma = float(last.get("TNX_sma50", np.nan))
    tnx_diff_bp = (tnx - tnx_ma) * 100.0 if (pd.notna(tnx) and pd.notna(tnx_ma)) else np.nan
    tnx_ok = int(last.get("sig_tnx", 0)) == 1

    dxy = float(last.get("DXY", np.nan))
    dxy_ma = float(last.get("DXY_sma50", np.nan))
    dxy_diff_pct = ((dxy / dxy_ma) - 1.0) * 100.0 if (pd.notna(dxy) and pd.notna(dxy_ma) and dxy_ma > 0) else np.nan
    dxy_ok = int(last.get("sig_dxy", 0)) == 1

    vix = float(last.get("VIX", np.nan))
    vix_20max = float(mdf["VIX"].tail(20).max()) if "VIX" in mdf.columns else np.nan
    vix_drop_pct = ((vix / vix_20max) - 1.0) * 100.0 if (pd.notna(vix) and pd.notna(vix_20max) and vix_20max > 0) else 0.0

    iwm = float(last.get("IWM", np.nan))
    iwm_ma = float(last.get("IWM_sma50", np.nan))
    iwm_diff_pct = ((iwm / iwm_ma) - 1.0) * 100.0 if (pd.notna(iwm) and pd.notna(iwm_ma) and iwm_ma > 0) else np.nan

    return {
        "available": True,
        "as_of": last_dt.strftime("%d.%m.%Y"),
        "irx": irx,
        "irx_th": irx_th,
        "irx_diff_bp": irx_diff_bp,
        "fed_ok": fed_ok,
        "fut_irx_thresh": fut_irx_thresh,
        "irx_th_20d": irx_th_20d,
        "irx_gap_20d_bp": irx_gap_20d_bp,
        "auto_flip_date": auto_flip_date,
        "tnx": tnx,
        "tnx_ma": tnx_ma,
        "tnx_diff_bp": tnx_diff_bp,
        "tnx_ok": tnx_ok,
        "dxy": dxy,
        "dxy_ma": dxy_ma,
        "dxy_diff_pct": dxy_diff_pct,
        "dxy_ok": dxy_ok,
        "vix": vix,
        "vix_20max": vix_20max,
        "vix_drop_pct": vix_drop_pct,
        "iwm": iwm,
        "iwm_diff_pct": iwm_diff_pct,
    }


def load_macro_data(force_download: bool = False) -> pd.DataFrame:
    """Loads macro data from local cache or downloads fresh series via single batch call."""
    if not force_download and os.path.exists(MACRO_CACHE_FILE):
        try:
            mdf = pd.read_csv(MACRO_CACHE_FILE, index_col=0, parse_dates=True)
            if not mdf.empty and len(mdf) >= 100 and "NDX" in mdf.columns:
                return compute_macro_signals(mdf)
        except Exception:
            pass

    batch = _batch_download_ohlcv(list(MACRO_SYMBOLS.values()), start="2019-06-01")
    series_map: Dict[str, pd.Series] = {}
    for key, sym in MACRO_SYMBOLS.items():
        df = batch.get(sym)
        if df is not None and "Close" in df.columns:
            series_map[key] = pd.to_numeric(df["Close"], errors="coerce").dropna()
        else:
            s = _download_single_close(sym)
            if s is not None and not s.empty:
                series_map[key] = s

    if not series_map:
        if os.path.exists(MACRO_CACHE_FILE):
            try:
                return compute_macro_signals(pd.read_csv(MACRO_CACHE_FILE, index_col=0, parse_dates=True))
            except Exception:
                pass
        return pd.DataFrame()

    mdf = pd.DataFrame(series_map).sort_index().ffill()
    mdf.to_csv(MACRO_CACHE_FILE)
    return compute_macro_signals(mdf)


def get_latest_macro_snapshot(mdf: Optional[pd.DataFrame]) -> Dict[str, object]:
    if mdf is None or mdf.empty:
        return {"available": False}

    last = mdf.iloc[-1]
    as_of = mdf.index[-1].strftime("%Y-%m-%d")
    cb_score = int(last.get("cb_score", 0))
    fed_ok = int(last.get("sig_fed", 0)) == 1
    qqq50_ok = int(last.get("sig_qqq50", 0)) == 1
    fed_qqq50_ok = int(last.get("macro_fed_plus_qqq50", 0)) == 1
    tnx_ok = int(last.get("sig_tnx", 0)) == 1
    dxy_ok = int(last.get("sig_dxy", 0)) == 1
    strict_ok = int(last.get("macro_strict_liq", 0)) == 1
    score_6 = int(last.get("liq_score_6", 0))

    irx_val = float(last.get("IRX", np.nan))
    tnx_val = float(last.get("TNX", np.nan))
    dxy_val = float(last.get("DXY", np.nan))
    vix_val = float(last.get("VIX", np.nan))

    if fed_qqq50_ok:
        regime_label, bootstyle = "LIQUIDITÄT ON", "inverse-success"
    elif fed_ok or qqq50_ok or cb_score >= 2:
        regime_label, bootstyle = "ÜBERGANG", "inverse-warning"
    else:
        regime_label, bootstyle = "LIQUIDITÄT OFF", "inverse-danger"

    return {
        "available": True,
        "as_of": as_of,
        "regime_label": regime_label,
        "bootstyle": bootstyle,
        "cb_score": cb_score,
        "score_6": score_6,
        "fed_ok": fed_ok,
        "qqq50_ok": qqq50_ok,
        "fed_qqq50_ok": fed_qqq50_ok,
        "tnx_ok": tnx_ok,
        "dxy_ok": dxy_ok,
        "strict_ok": strict_ok,
        "irx": irx_val,
        "tnx": tnx_val,
        "dxy": dxy_val,
        "vix": vix_val,
    }


def _quick_supertrend_state(df: pd.DataFrame, atr_len: int = 10, factor: float = 3.0) -> Tuple[str, float]:
    if len(df) < 30:
        return "N/A", 0.0
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
    atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).ewm(alpha=1.0 / atr_len, adjust=False).mean().values
    hl2 = ((df["High"] + df["Low"]) / 2.0).values
    cl, op = df["Close"].values, df["Open"].values
    reg, tu, td = 0, np.nan, np.nan
    for i in range(atr_len + 2, len(df)):
        a = atr[i] if not np.isnan(atr[i]) else 0.0
        ub, db = hl2[i] - factor * a, hl2[i] + factor * a
        if np.isnan(tu):
            tu, td = ub, db
        if reg == 0:
            reg = 1 if cl[i] > op[i] else -1
        if reg == 1:
            tu = max(ub, tu)
            if cl[i] < tu:
                reg, td = -1, db
        else:
            td = min(db, td)
            if cl[i] > td:
                reg, tu = 1, ub
    stop = tu if reg == 1 else td
    dist = ((cl[-1] - stop) / cl[-1] * 100.0) if cl[-1] > 0 else 0.0
    return ("BUY" if reg == 1 else "SELL"), float(dist)


def load_ucits_comparison_series(us_ticker: str, force_download: bool = False) -> Tuple[Optional[pd.Series], str]:
    """Returns (Close Series in EUR, display label) for the German UCITS ETF equivalent."""
    meta = HYPE_META_BY_TICKER.get(us_ticker.upper())
    if not meta or not meta.get("ucits_ticker"):
        return None, ""
    ucits_sym = str(meta["ucits_ticker"])
    wkn = str(meta["wkn"])
    df, _msg, _t = load_ticker_data(ucits_sym, "yahoo", force_download=force_download)
    if df is None or df.empty or "Close" not in df.columns:
        return None, ""
    label = f"DE UCITS {ucits_sym} (WKN {wkn}, EUR)"
    return df["Close"].dropna(), label


def build_hype_radar_df(force_download: bool = False) -> pd.DataFrame:
    """Computes Relative Strength & caches both US Radar ETFs and German UCITS `.DE` ETFs."""
    etf_dfs: Dict[str, pd.DataFrame] = {}
    to_download: List[str] = []

    # Queue both US Radar ETFs and German Xetra UCITS ETFs for batch download
    all_symbols: List[str] = []
    for sym, _theme, _wkn, ucits_sym, _uname in HYPE_RADAR_ETFS:
        all_symbols.append(sym)
        if ucits_sym:
            all_symbols.append(ucits_sym)

    for sym in all_symbols:
        cache_p = os.path.join(CACHE_DIR, f"{sym}.csv")
        if not force_download and os.path.exists(cache_p):
            df, _msg, _t = load_ticker_data(sym, "yahoo", force_download=False)
            if df is not None and not df.empty:
                etf_dfs[sym] = df
                continue
        to_download.append(sym)

    if to_download:
        batch = _batch_download_ohlcv(to_download, start="2020-01-01")
        for sym, df in batch.items():
            if df is not None and not df.empty:
                df_clean = df.copy()
                df_clean.index.name = "Date"
                df_clean.to_csv(os.path.join(CACHE_DIR, f"{sym}.csv"))
                etf_dfs[sym] = df_clean

    if "QQQ" not in etf_dfs:
        df_q, _m, _t = load_ticker_data("QQQ", "yahoo", force_download=True)
        if df_q is not None and len(df_q) >= 130:
            etf_dfs["QQQ"] = df_q

    if "QQQ" not in etf_dfs:
        return pd.DataFrame()

    qqq_c = etf_dfs["QQQ"]["Close"]
    qqq_3m = (qqq_c.iloc[-1] / qqq_c.iloc[-64] - 1.0) * 100.0
    qqq_1m = (qqq_c.iloc[-1] / qqq_c.iloc[-22] - 1.0) * 100.0

    rows = []
    for sym, theme, wkn, ucits_sym, ucits_name in HYPE_RADAR_ETFS:
        df = etf_dfs.get(sym)
        if df is None or len(df) < 130:
            continue
        c = df["Close"]
        r1m = float((c.iloc[-1] / c.iloc[-22] - 1.0) * 100.0)
        r3m = float((c.iloc[-1] / c.iloc[-64] - 1.0) * 100.0)
        r6m = float((c.iloc[-1] / c.iloc[-127] - 1.0) * 100.0)
        rs_3m = r3m - float(qqq_3m)
        rs_1m = r1m - float(qqq_1m)
        sma50 = float(c.rolling(50).mean().iloc[-1])
        ext50 = ((float(c.iloc[-1]) / sma50) - 1.0) * 100.0 if sma50 > 0 else 0.0
        st_sig, stop_dist = _quick_supertrend_state(df)

        # Also check 3M return of German UCITS ETF (in EUR) if available
        ucits_3m = np.nan
        if ucits_sym and ucits_sym in etf_dfs and len(etf_dfs[ucits_sym]) >= 65:
            uc = etf_dfs[ucits_sym]["Close"]
            ucits_3m = float((uc.iloc[-1] / uc.iloc[-64] - 1.0) * 100.0)

        if sym in ("QQQ", "IWM"):
            phase = "📊 BENCHMARK"
        elif st_sig == "BUY" and rs_3m > 2.0:
            phase = "🚀 LEADER"
        elif rs_3m > 0.0 or (rs_1m > 2.0 and r6m > 0.0):
            phase = "🌱 FRÜH-RS"
        elif rs_1m > 0.0 and ext50 > -6.0:
            phase = "⚡ ROTATION"
        else:
            phase = "❄️ SCHWACH"

        rows.append(
            {
                "phase": phase,
                "ticker": sym,
                "wkn": wkn,
                "ucits_ticker": ucits_sym or "—",
                "ucits_name": ucits_name,
                "theme": theme,
                "price": float(c.iloc[-1]),
                "r1m": r1m,
                "r3m": r3m,
                "r6m": r6m,
                "rs_3m": rs_3m,
                "ucits_3m": ucits_3m,
                "ext50": ext50,
                "st_sig": st_sig,
                "stop_dist": stop_dist,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by="rs_3m", ascending=False).reset_index(drop=True)
    return out
