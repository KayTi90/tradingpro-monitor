"""SEC EDGAR Dilution Radar & Cash Runway / Earnings Fundamentals for TradingPro v3.3.

Parses the actual SEC filing cover page (first 32 KB) to distinguish:
- Pure Corporate Debt / Bonds ("Senior Notes" -> 0% dilution, NO false alarm on GOOGL/NXPI!)
- Real Equity / ATM / Warrant / Convertible Offerings
- Exact Offering Size ($M or Share Count) and % Dilution relative to Market Cap / Shares Outstanding!
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
import time
import urllib.request
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple
import yfinance as yf

from .data_loader import CACHE_DIR, resolve_ticker


SEC_HEADERS = {"User-Agent": "TradingProAnalytics/3.3 (local-desktop-research@localhost)"}
SEC_CIK_CACHE = os.path.join(CACHE_DIR, "_SEC_CIK_MAP.json")

ACUTE_OFFERING_FORMS = {"424B5", "424B3", "424B2", "424B7", "S-1", "F-1"}
SHELF_FORMS = {"S-3", "S-3ASR", "F-3", "EFFECT"}
ALL_DILUTION_FORMS = ACUTE_OFFERING_FORMS | SHELF_FORMS

_cik_map_lock = threading.Lock()
_cik_map_cache: Optional[Dict[str, str]] = None


def get_sec_cik_map(force_refresh: bool = False) -> Dict[str, str]:
    global _cik_map_cache
    with _cik_map_lock:
        if _cik_map_cache is not None and not force_refresh:
            return _cik_map_cache

        if not force_refresh and os.path.exists(SEC_CIK_CACHE):
            age_days = (time.time() - os.path.getmtime(SEC_CIK_CACHE)) / 86400.0
            if age_days < 14:
                try:
                    with open(SEC_CIK_CACHE, "r", encoding="utf-8") as f:
                        _cik_map_cache = json.load(f)
                        return _cik_map_cache
                except Exception:
                    pass

        try:
            req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            mapping = {str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}
            with open(SEC_CIK_CACHE, "w", encoding="utf-8") as f:
                json.dump(mapping, f)
            _cik_map_cache = mapping
            return mapping
        except Exception:
            if os.path.exists(SEC_CIK_CACHE):
                try:
                    with open(SEC_CIK_CACHE, "r", encoding="utf-8") as f:
                        _cik_map_cache = json.load(f)
                        return _cik_map_cache
                except Exception:
                    pass
            return {}


def _fetch_filing_cover_text(cik: str, accession: str, primary_doc: str, max_bytes: int = 32768) -> str:
    if not cik or not accession or not primary_doc:
        return ""
    cik_int = str(int(cik))
    acc_clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{primary_doc}"
    try:
        req = urllib.request.Request(url, headers={**SEC_HEADERS, "Range": f"bytes=0-{max_bytes}"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        try:
            req = urllib.request.Request(url, headers=SEC_HEADERS)
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw = resp.read(max_bytes).decode("utf-8", errors="ignore")
        except Exception:
            return ""
    txt = re.sub(r"<[^>]+>", " ", raw)
    txt = html.unescape(re.sub(r"\s+", " ", txt)).strip()
    return txt[:4000]


def _parse_offering_cover(cover_text: str, form: str) -> Dict[str, object]:
    """Classifies security type (DEBT vs ATM vs EQUITY vs CONVERTIBLE) and extracts size ($M / shares)."""
    if not cover_text:
        return {"sec_type": "UNKNOWN", "size_usd_m": None, "shares_m": None}

    head = cover_text[:2000]
    low = head.lower()

    # 1. Classify Security Type
    has_convertible = bool(re.search(r"\b(convertible|exchangeable)\s+(senior\s+)?(notes|debentures|bonds)", low))
    has_atm = bool(
        re.search(
            r"(equity distribution agreement|at-the-market|sales agreement|controlled equity offering|open market sale agreement)",
            low,
        )
    )
    has_equity_words = bool(
        re.search(
            r"\b(common stock|ordinary shares|class [abc] common|american depositary shares|pre-funded warrants|warrants to purchase|shares of common)\b",
            low,
        )
    )
    has_debt_words = bool(
        re.search(
            r"\b(senior notes|notes due 20\d\d|fixed rate notes|floating rate notes|aggregate principal amount of|debentures due|subordinated notes)\b",
            low,
        )
    )

    if has_convertible:
        sec_type = "CONVERTIBLE"
    elif has_atm and (has_equity_words or not has_debt_words):
        sec_type = "ATM"
    elif has_debt_words and not has_equity_words:
        sec_type = "DEBT"
    elif has_equity_words:
        sec_type = "EQUITY"
    elif has_debt_words:
        sec_type = "DEBT"
    else:
        sec_type = "SHELF" if form in SHELF_FORMS else "UNKNOWN"

    # 2. Extract Dollar Offering Size (in USD Millions; ignore par value like $0.0001)
    size_usd_m: Optional[float] = None

    # Pattern A: "$1,000,000,000" or "$225,000,000" or "€1,000,000,000"
    m_exact = re.findall(r"(?:US)?[\$€£]\s*([0-9]{1,3}(?:,[0-9]{3}){2,})(?:\.\d+)?", head)
    if m_exact:
        vals = [float(x.replace(",", "")) / 1e6 for x in m_exact if float(x.replace(",", "")) >= 1_000_000]
        if vals:
            size_usd_m = round(vals[0], 1)

    # Pattern B: "$250.5 million" or "$1.2 billion"
    if size_usd_m is None:
        m_word = re.search(r"[\$€£]\s*([0-9]+(?:\.[0-9]+)?)\s*(million|billion)\b", low)
        if m_word:
            num = float(m_word.group(1))
            mult = 1000.0 if m_word.group(2) == "billion" else 1.0
            if num * mult >= 1.0:
                size_usd_m = round(num * mult, 1)

    # 3. Extract Share Count (in Millions of Shares)
    shares_m: Optional[float] = None
    m_sh = re.findall(
        r"\b([0-9]{1,3}(?:,[0-9]{3})+)\s+(?:shares\b|class\s+[abc]\s+common|american depositary shares|ordinary shares)",
        low,
    )
    if not m_sh:
        m_sh = re.findall(r"(?:purchase|offering|issuing|up to)\s+([0-9]{1,3}(?:,[0-9]{3})+)\s+shares", low)
    if m_sh:
        s_vals = [float(x.replace(",", "")) / 1e6 for x in m_sh if float(x.replace(",", "")) >= 10_000]
        if s_vals:
            shares_m = round(max(s_vals), 3)

    return {"sec_type": sec_type, "size_usd_m": size_usd_m, "shares_m": shares_m}


def fetch_sec_historical_runway(cik: str) -> List[Dict[str, object]]:
    """Fetches quarterly unrestricted Cash+ST Investments, OCF, CapEx, and FCF Runway
    from SEC EDGAR XBRL companyfacts so we know the exact runway at each past offering!
    """
    if not cik:
        return []
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        req = urllib.request.Request(url, headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            us_gaap = json.loads(resp.read().decode("utf-8")).get("facts", {}).get("us-gaap", {})
    except Exception:
        return []

    def _instant(tags: List[str]) -> Dict[str, float]:
        res: Dict[str, float] = {}
        for tag in tags:
            for item in us_gaap.get(tag, {}).get("units", {}).get("USD", []):
                if item.get("form") in ("10-Q", "10-K") and item.get("end") and "start" not in item:
                    end = str(item["end"])
                    if end not in res:
                        res[end] = float(item["val"])
        return res

    def _flow_ann(tags: List[str]) -> Dict[str, float]:
        res: Dict[str, Tuple[int, float]] = {}
        for tag in tags:
            for item in us_gaap.get(tag, {}).get("units", {}).get("USD", []):
                if item.get("form") in ("10-Q", "10-K") and item.get("start") and item.get("end"):
                    try:
                        d1 = datetime.strptime(str(item["start"]), "%Y-%m-%d")
                        d2 = datetime.strptime(str(item["end"]), "%Y-%m-%d")
                        days = (d2 - d1).days
                    except Exception:
                        continue
                    if 70 <= days <= 380:
                        end = str(item["end"])
                        ann = float(item["val"]) * (365.25 / max(days, 85))
                        prev = res.get(end)
                        if prev is None or days > prev[0]:
                            res[end] = (days, ann)
        return {k: v[1] for k, v in res.items()}

    combined = _instant(
        ["CashCashEquivalentsAndShortTermInvestments", "CashAndCashEquivalentsAndShortTermInvestments"]
    )
    cash_only = _instant(["CashAndCashEquivalentsAtCarryingValue"])
    st_inv = _instant(
        [
            "ShortTermInvestments",
            "MarketableSecuritiesCurrent",
            "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
            "DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent",
            "AvailableForSaleSecuritiesCurrent",
        ]
    )
    cash_restr = _instant(["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"])

    ocf = _flow_ann(
        [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ]
    )
    capex = _flow_ann(
        [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
        ]
    )

    all_ends = sorted(set(combined.keys()) | set(cash_only.keys()) | set(cash_restr.keys()))
    history: List[Dict[str, object]] = []
    for end in all_ends:
        if end < "2021-01-01" or end not in ocf:
            continue
        if end in combined:
            liq = combined[end]
        elif end in cash_only:
            liq = cash_only[end] + st_inv.get(end, 0.0)
        else:
            liq = cash_restr[end]
        c_m = liq / 1e6
        ocf_m = ocf[end] / 1e6
        cap_m = abs(capex.get(end, 0.0)) / 1e6
        fcf_m = ocf_m - cap_m
        if fcf_m >= 0:
            rw = 999.0
        elif c_m > 0:
            rw = round((c_m / abs(fcf_m)) * 12.0, 1)
        else:
            rw = 0.0
        history.append(
            {
                "end": end,
                "cash_m": round(c_m, 1),
                "ocf_m": round(ocf_m, 1),
                "capex_m": round(cap_m, 1),
                "fcf_m": round(fcf_m, 1),
                "runway_m": rw,
            }
        )
    return history


def _lookup_runway_before_date(
    history: List[Dict[str, object]], filing_date: str
) -> Optional[Dict[str, object]]:
    """Returns the latest quarterly balance-sheet snapshot strictly BEFORE the offering date."""
    best = None
    for row in history:
        if str(row["end"]) < filing_date:
            best = row
        else:
            break
    return best


def fetch_sec_dilution_filings(
    ticker: str,
    lookback_days: int = 730,
    market_cap_m: float = -1.0,
    shares_out_m: float = -1.0,
    last_price: float = -1.0,
    max_cover_inspections: int = 4,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Fetches SEC filings, inspects cover pages, and attaches historical FCF Runway at each offering."""
    sym = resolve_ticker(ticker)
    if "." in sym or "-USD" in sym:
        return [], []

    cik_map = get_sec_cik_map()
    cik = cik_map.get(sym)
    if not cik:
        return [], []

    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        req = urllib.request.Request(url, headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])

        out: List[Dict[str, object]] = []
        seen = set()
        inspected = 0

        for form, fdate, acc, doc in zip(forms, dates, accs, docs):
            if fdate < cutoff:
                continue
            if form not in ALL_DILUTION_FORMS:
                continue
            key = (form, fdate)
            if key in seen:
                continue
            seen.add(key)

            entry: Dict[str, object] = {
                "form": str(form),
                "date": str(fdate),
                "sec_type": "SHELF" if form in SHELF_FORMS else "UNKNOWN",
                "size_usd_m": None,
                "shares_m": None,
                "dilution_pct": None,
                "runway_then_m": None,
                "cash_then_m": None,
                "fcf_then_m": None,
            }

            if form in ACUTE_OFFERING_FORMS and inspected < max_cover_inspections:
                cover = _fetch_filing_cover_text(cik, str(acc), str(doc))
                parsed = _parse_offering_cover(cover, str(form))
                entry.update(parsed)
                inspected += 1

                if entry["sec_type"] != "DEBT":
                    dil_pct = None
                    if entry["size_usd_m"] is not None and market_cap_m > 0:
                        dil_pct = (float(entry["size_usd_m"]) / market_cap_m) * 100.0
                    elif entry["shares_m"] is not None and shares_out_m > 0:
                        dil_pct = (float(entry["shares_m"]) / shares_out_m) * 100.0
                        if entry["size_usd_m"] is None and last_price > 0:
                            entry["size_usd_m"] = round(float(entry["shares_m"]) * last_price, 1)
                    if dil_pct is not None:
                        entry["dilution_pct"] = round(min(dil_pct, 999.0), 2)

            out.append(entry)
            if len(out) >= 10:
                break

        # Fetch historical quarterly FCF runway ONLY if there are real non-debt offerings
        has_real_eq = any(
            fl["form"] in ACUTE_OFFERING_FORMS and fl.get("sec_type") != "DEBT" for fl in out
        )
        runway_hist = fetch_sec_historical_runway(cik) if has_real_eq else []
        if runway_hist:
            for fl in out:
                if fl["form"] in ACUTE_OFFERING_FORMS and fl.get("sec_type") != "DEBT":
                    snap = _lookup_runway_before_date(runway_hist, str(fl["date"]))
                    if snap:
                        fl["runway_then_m"] = snap["runway_m"]
                        fl["cash_then_m"] = snap["cash_m"]
                        fl["fcf_then_m"] = snap["fcf_m"]

        return out, runway_hist
    except Exception:
        return [], []


def _extract_next_earnings(tk: yf.Ticker) -> str:
    try:
        cal = tk.calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if isinstance(ed, list) and ed:
                dt = ed[0]
                if hasattr(dt, "strftime"):
                    return dt.strftime("%Y-%m-%d")
                return str(dt)[:10]
            if hasattr(ed, "strftime"):
                return ed.strftime("%Y-%m-%d")
    except Exception:
        pass
    return ""


def fetch_ticker_fundamentals(ticker: str) -> Dict[str, object]:
    """Fetches combined SEC cover-parsed dilution filings + Yahoo Cash Runway / MCap / Earnings."""
    sym = resolve_ticker(ticker)
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M")

    if "-USD" in sym:
        return {
            "ticker": sym,
            "updated_at": now_str,
            "risk_level": -1,
            "risk_badge": "—",
            "total_cash_m": -1.0,
            "ocf_m": 0.0,
            "runway_months": -1.0,
            "short_pct": -1.0,
            "next_earnings": "",
            "last_offering_date": "",
            "last_offering_form": "",
            "has_active_shelf": 0,
            "filings_json": "[]",
        }

    # Fetch Yahoo Finance info FIRST so we have Market Cap & Shares Outstanding for % dilution!
    total_cash_m = -1.0
    ocf_m = 0.0
    runway_months = -1.0
    short_pct = -1.0
    next_earnings = ""
    market_cap_m = -1.0
    shares_out_m = -1.0
    last_price = -1.0

    try:
        tk = yf.Ticker(sym)
        info = tk.info or {}
        cash = info.get("totalCash")
        ocf = info.get("operatingCashflow")
        mcap = info.get("marketCap")
        sh_out = info.get("sharesOutstanding")
        px = info.get("currentPrice") or info.get("regularMarketPrice")

        if cash is not None:
            total_cash_m = float(cash) / 1e6
        if ocf is not None:
            ocf_m = float(ocf) / 1e6
        if mcap is not None and float(mcap) > 0:
            market_cap_m = float(mcap) / 1e6
        if sh_out is not None and float(sh_out) > 0:
            shares_out_m = float(sh_out) / 1e6
        if px is not None and float(px) > 0:
            last_price = float(px)

        if ocf is not None and ocf >= 0:
            runway_months = 999.0
        elif ocf is not None and ocf < 0 and cash is not None and cash > 0:
            runway_months = round((float(cash) / abs(float(ocf))) * 12.0, 1)
        elif ocf is not None and ocf < 0 and (cash is None or cash <= 0):
            runway_months = 0.0

        sp = info.get("shortPercentOfFloat")
        if sp is not None:
            short_pct = round(float(sp) * 100.0, 1)

        next_earnings = _extract_next_earnings(tk)
    except Exception:
        pass

    filings, runway_hist = fetch_sec_dilution_filings(
        sym,
        lookback_days=730,
        market_cap_m=market_cap_m,
        shares_out_m=shares_out_m,
        last_price=last_price,
    )

    # Upgrade current Runway & Cash using true FCF-Runway (OCF - CapEx) from latest SEC XBRL quarter!
    if runway_hist:
        latest_q = runway_hist[-1]
        total_cash_m = float(latest_q["cash_m"])
        ocf_m = float(latest_q["fcf_m"])  # True FCF burn (OCF - CapEx)
        runway_months = float(latest_q["runway_m"])

    # Find the most impactful REAL equity/ATM/convertible offering in the recent 30d / 90d windows
    # (so a tiny 0.2% filing one day later never masks a $400M offering from the day before!)
    recent_eq_30: List[Tuple[int, Dict[str, object]]] = []
    recent_eq_90: List[Tuple[int, Dict[str, object]]] = []
    first_eq_any: Optional[Tuple[int, Dict[str, object]]] = None
    has_shelf = 0

    for fl in filings:
        f_form = str(fl["form"])
        f_type = str(fl.get("sec_type", "UNKNOWN"))
        f_date = str(fl["date"])
        if f_form in SHELF_FORMS:
            has_shelf = 1
        if f_form in ACUTE_OFFERING_FORMS and f_type != "DEBT":
            d_ago = 9999
            try:
                d_ago = max(0, (now - datetime.strptime(f_date, "%Y-%m-%d")).days)
            except Exception:
                pass
            if first_eq_any is None:
                first_eq_any = (d_ago, fl)
            if d_ago <= 30:
                recent_eq_30.append((d_ago, fl))
            if d_ago <= 90:
                recent_eq_90.append((d_ago, fl))

    def _pick_largest(cands: List[Tuple[int, Dict[str, object]]]) -> Tuple[int, Dict[str, object]]:
        return max(cands, key=lambda x: float(x[1].get("dilution_pct") or 0.0))

    if recent_eq_30:
        days_since_off, last_eq_off = _pick_largest(recent_eq_30)
    elif recent_eq_90:
        days_since_off, last_eq_off = _pick_largest(recent_eq_90)
    elif first_eq_any is not None:
        days_since_off, last_eq_off = first_eq_any
    else:
        days_since_off, last_eq_off = 9999, None

    last_off_date = str(last_eq_off["date"]) if last_eq_off else ""
    last_off_form = str(last_eq_off["form"]) if last_eq_off else ""
    last_off_type = str(last_eq_off.get("sec_type", "")) if last_eq_off else ""
    last_off_dil = last_eq_off.get("dilution_pct") if last_eq_off else None
    last_off_size = last_eq_off.get("size_usd_m") if last_eq_off else None

    # Size-Aware Severity Classification!
    # Mini offerings (< 2% MCap, e.g. ASTS 0.2% partner share issuance) do NOT trigger red alarm!
    is_major_off = (last_off_dil is None) or (float(last_off_dil) >= 5.0)
    is_mini_off = (last_off_dil is not None) and (float(last_off_dil) < 2.0)

    def _short_type(t: str) -> str:
        return {"ATM": "ATM", "CONVERTIBLE": "CONV", "EQUITY": "OFF"}.get(t, "OFF")

    dil_tag = f" {float(last_off_dil):.0f}%" if last_off_dil is not None else ""

    if last_eq_off and days_since_off <= 30 and not is_mini_off:
        risk_level = 4 if is_major_off else 3
        icon = "🚨" if is_major_off else "🟠"
        risk_badge = f"{icon} {_short_type(last_off_type)}{dil_tag} ({days_since_off}T)"
    elif 0.0 <= runway_months < 12.0:
        risk_level = 3
        risk_badge = f"🔴 CASH {runway_months:.0f}M"
    elif last_eq_off and days_since_off <= 90 and not is_mini_off:
        risk_level = 3 if (last_off_dil is not None and float(last_off_dil) >= 10.0) else 2
        icon = "🚨" if risk_level == 3 else "🟠"
        risk_badge = f"{icon} {_short_type(last_off_type)}{dil_tag} ({days_since_off}T)"
    elif 12.0 <= runway_months < 24.0:
        risk_level = 2
        risk_badge = f"🟠 CASH {runway_months:.0f}M"
    elif last_eq_off and days_since_off <= 30 and is_mini_off:
        risk_level = 1
        risk_badge = f"🟡 MINI {float(last_off_dil):.1f}%"
    elif has_shelf and 0.0 <= runway_months < 48.0:
        risk_level = 1
        risk_badge = "🟡 SHELF"
    elif runway_months >= 24.0 or total_cash_m >= 0 or filings:
        risk_level = 0
        risk_badge = "🟢 SAUBER"
    else:
        risk_level = -1
        risk_badge = "—"

    return {
        "ticker": sym,
        "updated_at": now_str,
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "total_cash_m": round(total_cash_m, 1) if total_cash_m >= 0 else -1.0,
        "ocf_m": round(ocf_m, 1),
        "runway_months": runway_months,
        "short_pct": short_pct,
        "next_earnings": next_earnings,
        "last_offering_date": last_off_date,
        "last_offering_form": f"{last_off_form}/{last_off_type}" if last_off_type else last_off_form,
        "has_active_shelf": has_shelf,
        "filings_json": json.dumps(filings),
    }


def fetch_fundamentals_batch(
    tickers: List[str],
    cancel_event: Optional[threading.Event] = None,
    on_item_done: Optional[Callable[[int, int, str, Dict[str, object]], None]] = None,
) -> List[Dict[str, object]]:
    get_sec_cik_map()
    total = len(tickers)
    out: List[Dict[str, object]] = []
    for idx, t in enumerate(tickers, 1):
        if cancel_event is not None and cancel_event.is_set():
            break
        rec = fetch_ticker_fundamentals(t)
        out.append(rec)
        if on_item_done is not None:
            on_item_done(idx, total, t, rec)
        time.sleep(0.20)
    return out


def format_filing_summary(fl: Dict[str, object]) -> str:
    """Formats one SEC filing cleanly with type, dollar/share volume, and % Market Cap dilution."""
    form = str(fl.get("form", ""))
    fdate = str(fl.get("date", ""))
    stype = str(fl.get("sec_type", "UNKNOWN"))
    sz = fl.get("size_usd_m")
    sh = fl.get("shares_m")
    dil = fl.get("dilution_pct")

    type_labels = {
        "DEBT": "ANLEIHE (0% Verw.)",
        "ATM": "ATM-Aktien",
        "EQUITY": "Aktien/Warrants",
        "CONVERTIBLE": "Wandelanleihe",
        "SHELF": "Shelf-Rahmen",
        "UNKNOWN": "Prospekt",
    }
    lbl = type_labels.get(stype, stype)
    parts = [f"{fdate} | {form:<6} | {lbl}"]

    if sz is not None:
        sz_f = float(sz)
        sz_s = f"${sz_f/1000:.2f}B" if sz_f >= 1000 else f"${sz_f:.1f}M"
        parts.append(sz_s)
    elif sh is not None:
        parts.append(f"{float(sh):.2f}M Stk")

    if stype != "DEBT" and dil is not None:
        parts.append(f"≈ {float(dil):.1f}% MCap")

    rw_then = fl.get("runway_then_m")
    if stype != "DEBT" and rw_then is not None:
        rw_f = float(rw_then)
        if rw_f >= 900:
            parts.append("Runway damals: CF+")
        else:
            c_then = fl.get("cash_then_m")
            c_s = f", Cash ${float(c_then):.0f}M" if c_then is not None else ""
            parts.append(f"Runway damals: {rw_f:.1f}M{c_s}")

    return " | ".join(parts)


def _build_dilution_trigger_summary(filings: List[Dict[str, object]], current_runway: float) -> str:
    """Summarizes the company's historical 'pain threshold' (Runway at which it triggered KEs)."""
    rws = [
        float(fl["runway_then_m"])
        for fl in filings
        if fl.get("sec_type") != "DEBT"
        and fl.get("runway_then_m") is not None
        and float(fl["runway_then_m"]) < 900.0
    ]
    if not rws:
        return ""

    min_rw = min(rws)
    avg_rw = sum(rws) / len(rws)
    cur_s = f"{current_runway:.1f}M" if 0 <= current_runway < 900 else "CF+"

    if avg_rw <= 12.0:
        profile = "⚠️ NOT-VERWÄSSERER (druckt meist erst bei <12M Runway)"
    elif avg_rw <= 24.0:
        profile = "🔔 SCHWELLEN-VERWÄSSERER (Schmerzgrenze ~12–24M Runway)"
    else:
        profile = "🎯 OPPORTUNISTISCH (druckt 'Sell into Strength' auch bei >24M Cash!)"

    proximity = ""
    if 0 <= current_runway <= 24.0 and current_runway <= max(min_rw * 1.25, 12.0):
        proximity = "  -> 🚨 AKTUELL NAHE SCHMERZGRENZE!"
    elif 0 <= current_runway < 900 and avg_rw <= 24.0:
        buffer = current_runway - min_rw
        if buffer > 0:
            proximity = f"  (Puffer bis Hist.-Tief: +{buffer:.1f}M)"

    return (
        f"  KE-AUSLÖSEPROFIL: {profile}\n"
        f"    Historischer FCF-Runway bei KE: Min {min_rw:.1f}M | Ø {avg_rw:.1f}M (n={len(rws)}) | Aktuell: {cur_s}{proximity}"
    )


def format_fundamentals_report_block(fund: Optional[Dict[str, object]]) -> str:
    if not fund or int(fund.get("risk_level", -1)) == -1:
        return ""

    cash_m = float(fund.get("total_cash_m", -1.0))
    ocf_m = float(fund.get("ocf_m", 0.0))
    rw = float(fund.get("runway_months", -1.0))
    sp = float(fund.get("short_pct", -1.0))
    ne = str(fund.get("next_earnings") or "")
    badge = str(fund.get("risk_badge") or "—")

    cash_s = f"${cash_m:,.0f}M" if cash_m >= 0 else "N/A"
    ocf_s = f"${ocf_m:+,.0f}M/J" if cash_m >= 0 else "N/A"
    if rw >= 900:
        rw_s = "Profitabel (CF+)"
    elif rw >= 0:
        rw_s = f"{rw:.1f} Monate"
    else:
        rw_s = "N/A"
    short_s = f"{sp:.1f}%" if sp >= 0 else "N/A"

    earn_s = "N/A"
    if ne:
        try:
            d_left = (datetime.strptime(ne, "%Y-%m-%d").date() - datetime.now().date()).days
            earn_s = f"{ne} (in {d_left}T)" if d_left >= 0 else ne
        except Exception:
            earn_s = ne

    filings = []
    try:
        filings = json.loads(str(fund.get("filings_json") or "[]"))
    except Exception:
        filings = []

    lines = [
        "-" * 72,
        f"DILUTION & FUNDAMENTALS ({badge})  [Stand: {fund.get('updated_at', '')}]",
        f"  Cash+ST: {cash_s} | FCF-Burn: {ocf_s} | FCF-Runway: {rw_s} | Short: {short_s} | Earnings: {earn_s}",
    ]
    trig_summary = _build_dilution_trigger_summary(filings, rw)
    if trig_summary:
        lines.append(trig_summary)

    if filings:
        lines.append("  SEC-FILINGS (Typ, Volumen, % MCap & FCF-Runway zum Zeitpunkt der KE):")
        for fl in filings[:6]:
            lines.append(f"    • {format_filing_summary(fl)}")
    else:
        lines.append("  SEC-Filings (2J): Keine Offering-/Shelf-Filings gefunden.")

    return "\n".join(lines)


def fetch_10q_atm_report(ticker: str) -> Dict[str, object]:
    """Downloads the latest SEC 10-Q/10-K for a US ticker and extracts ATM usage,
    remaining capacity, and cover-page share count vs quarter-end share count.
    """
    sym = resolve_ticker(ticker)
    if "." in sym or "-USD" in sym:
        return {"ok": False, "error": f"{sym} ist keine US-SEC-Aktie (kein 10-Q/10-K verfügbar)."}

    cik_map = get_sec_cik_map()
    cik = cik_map.get(sym)
    if not cik:
        return {"ok": False, "error": f"Kein SEC-CIK-Eintrag für {sym} gefunden."}

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        req = urllib.request.Request(url, headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"SEC-Abruf fehlgeschlagen: {exc}"}

    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    rep_dates = recent.get("reportDate", [])
    accs = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    target = None
    for form, fdate, rdate, acc, doc in zip(forms, dates, rep_dates, accs, docs):
        if form in ("10-Q", "10-K"):
            target = (str(form), str(fdate), str(rdate), str(acc), str(doc))
            break

    if not target:
        return {"ok": False, "error": f"Kein 10-Q oder 10-K für {sym} gefunden (evtl. Foreign Private Issuer 20-F/6-K)."}

    form, fdate, rdate, acc, doc = target
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"

    try:
        req_doc = urllib.request.Request(doc_url, headers=SEC_HEADERS)
        with urllib.request.urlopen(req_doc, timeout=25) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return {"ok": False, "url": doc_url, "error": f"Download von {form} ({fdate}) fehlgeschlagen: {exc}"}

    txt = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw))).strip()

    # 1. Cover-page shares outstanding (latest share count as of filing date!)
    cover_head = txt[:30000]
    cover_shares_info = "Nicht auf Deckblatt erkannt"
    m_cov = re.search(
        r"([0-9]{1,3}(?:,[0-9]{3}){2,})\s+shares\s+of\s+(?:[A-Za-z0-9\-\s]{0,35})?common\s+stock\s+outstanding\s+as\s+of\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        cover_head,
        re.I,
    )
    if not m_cov:
        m_cov = re.search(
            r"as\s+of\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})[^.]{0,80}?([0-9]{1,3}(?:,[0-9]{3}){2,})\s+shares",
            cover_head,
            re.I,
        )
        if m_cov:
            cover_shares_info = f"{m_cov.group(2)} Aktien (Stand Deckblatt: {m_cov.group(1)})"
    else:
        cover_shares_info = f"{m_cov.group(1)} Aktien (Stand Deckblatt: {m_cov.group(2)})"

    # 2. Extract high-signal ATM / Sales Agreement / Remaining Capacity excerpts
    pattern = re.compile(
        r"(Equity Distribution Agreement|Sales Agreement|At-the-Market|ATM Program|ATM Offering|Open Market Sale Agreement|remaining available for (?:sale|issuance))",
        re.I,
    )
    candidates: List[Tuple[int, int, str]] = []
    seen_starts: List[int] = []
    for m in pattern.finditer(txt):
        s = max(0, m.start() - 120)
        if any(abs(s - ps) < 420 for ps in seen_starts):
            continue
        e = min(len(txt), m.end() + 520)
        snippet = txt[s:e].strip()
        low = snippet.lower()
        # Score snippet by concrete numerical/status relevance
        score = 0
        if "$" in snippet:
            score += 3
        if any(w in low for w in ("remaining", "available", "ongoing", "completed", "terminated")):
            score += 4
        if any(w in low for w in ("gross proceeds", "net proceeds", "average", "shares issued", "sold")):
            score += 3
        if any(w in low for w in ("million", "billion", "aggregate")):
            score += 2
        if "table of contents" in low and "$" not in snippet:
            score -= 5
        if score >= 2:
            seen_starts.append(s)
            candidates.append((score, m.start(), snippet))

    # Sort best-scored first, then keep original document order for readability
    top_cands = sorted(candidates, key=lambda x: (-x[0], x[1]))[:6]
    top_cands.sort(key=lambda x: x[1])

    # 3. Quick regex highlights for explicit "remaining available" dollar values
    rem_highlights = []
    for _, _, snip in top_cands:
        for mr in re.finditer(
            r"((?:approximately\s+)?\$[0-9,\.]+\s*(?:million|billion|thousand)?[^.]{0,90}?(?:remaining|remained\s+available|available\s+for\s+(?:future\s+)?(?:sale|issuance)))",
            snip,
            re.I,
        ):
            rem_highlights.append(mr.group(1).strip())

    # Format clean text report
    lines = [
        f"📄 SEC {form} ATM- & VERWÄSSERUNGS-CHECK: {sym}",
        f"Eingereicht: {fdate}  |  Berichtsstichtag: {rdate}",
        f"Aktienstand Deckblatt: {cover_shares_info}",
        f"Original-Dokument: {doc_url}",
        "=" * 74,
    ]
    if rem_highlights:
        lines.append("⚡ DIREKT ERKANNTER OFFENER RESTBETRAG:")
        for rh in list(dict.fromkeys(rem_highlights))[:3]:
            lines.append(f"  → {rh}")
        lines.append("-" * 74)

    if top_cands:
        lines.append(f"RELEVANTE ATM- / OFFERING-TEXTSTELLEN AUS {form} ({len(top_cands)} Auszüge):")
        for idx, (_sc, _pos, snip) in enumerate(top_cands, 1):
            lines.append(f"\n[{idx}] ...{snip}...")
    else:
        lines.append(f"Keine aktiven ATM- / Equity-Distribution-Hinweise im neuesten {form} ({fdate}) gefunden.")

    return {
        "ok": True,
        "ticker": sym,
        "form": form,
        "filing_date": fdate,
        "report_date": rdate,
        "url": doc_url,
        "cover_shares": cover_shares_info,
        "excerpts_count": len(top_cands),
        "report_text": "\n".join(lines),
    }
