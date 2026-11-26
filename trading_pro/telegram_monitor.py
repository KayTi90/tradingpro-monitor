"""Headless TradingPro Monitor: Core R1-F, Macro-Liquidity & 128-Stock Supertrend
with Telegram Push Alerts and Mobile PWA Dashboard (`docs/index.html`).

Designed to run both:
- Autonomously in the cloud via GitHub Actions (Mon-Fri after US market close, 0 EUR)
- Locally on Windows (for instant testing or local sync with `trading_pro_v3.db`)
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import download_tickers_parallel, load_ticker_data
from .macro import (
    get_core_allocation_status,
    get_latest_macro_snapshot,
    get_macro_flip_diagnostics,
    load_macro_data,
)
from .strategy import DEFAULT_STRATEGY_CONFIG, calculate_position_size, run_strategy


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCHLIST_FILE = os.path.join(PROJECT_ROOT, "watchlist.json")
STATE_FILE = os.path.join(PROJECT_ROOT, "monitor_state.json")
LOCAL_TG_CONFIG = os.path.join(PROJECT_ROOT, "telegram_config.json")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
DASHBOARD_HTML = os.path.join(DOCS_DIR, "index.html")
DASHBOARD_MANIFEST = os.path.join(DOCS_DIR, "manifest.json")


def sync_watchlist_from_local_db(watchlist_path: str = WATCHLIST_FILE) -> Optional[Dict[str, Any]]:
    """Exports Watchlist, Depot & Strategy Settings from local `trading_pro_v3.db` to `watchlist.json`."""
    try:
        from . import database as db

        if not os.path.exists(db.DB_FILE):
            return None
        db.init_db()
        df = db.get_all_metrics_df()
        if df.empty:
            return None
        tickers = [
            {"ticker": str(r["ticker"]), "name": str(r.get("company_name") or r["ticker"])}
            for _, r in df.sort_values("ticker").iterrows()
        ]
        depot = sorted(db.get_depot_tickers())
        settings = db.load_settings()
        payload = {
            "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tickers": tickers,
            "depot": depot,
            "settings": settings,
        }
        with open(watchlist_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return payload
    except Exception:
        return None


def load_watchlist_config(watchlist_path: str = WATCHLIST_FILE) -> Dict[str, Any]:
    if not os.path.exists(watchlist_path):
        synced = sync_watchlist_from_local_db(watchlist_path)
        if synced is not None:
            return synced
        raise FileNotFoundError(
            f"Weder {watchlist_path} noch eine lokale trading_pro_v3.db gefunden."
        )
    with open(watchlist_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    settings = dict(DEFAULT_STRATEGY_CONFIG)
    settings.update(data.get("settings") or {})
    data["settings"] = settings
    data["depot"] = list(data.get("depot") or [])
    data["tickers"] = list(data.get("tickers") or [])
    return data


def load_monitor_state(state_path: str = STATE_FILE) -> Dict[str, Any]:
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_monitor_state(state: Dict[str, Any], state_path: str = STATE_FILE) -> None:
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)


def load_local_cloud_config(cfg_path: str = LOCAL_TG_CONFIG) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "bot_token": "",
        "chat_id": "",
        "dashboard_url": "",
        "github_repo": "",
        "github_branch": "main",
        "github_token": "",
        "auto_sync_github": True,
    }
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                defaults.update(raw)
        except Exception:
            pass
    return defaults


def save_local_cloud_config(cfg: Dict[str, Any], cfg_path: str = LOCAL_TG_CONFIG) -> None:
    merged = load_local_cloud_config(cfg_path)
    merged.update(cfg)
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, ensure_ascii=False)


def get_telegram_credentials() -> Tuple[str, str, str]:
    """Reads Telegram Bot Token, Chat ID & optional Dashboard URL from env vars or local config."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    dash_url = os.environ.get("DASHBOARD_URL", "").strip()
    if not token or not chat_id or not dash_url:
        cfg = load_local_cloud_config()
        token = token or str(cfg.get("bot_token", "")).strip()
        chat_id = chat_id or str(cfg.get("chat_id", "")).strip()
        dash_url = dash_url or str(cfg.get("dashboard_url", "")).strip()
    return token, chat_id, dash_url


def _github_put_file(
    repo: str,
    branch: str,
    token: str,
    rel_path: str,
    content_bytes: bytes,
    commit_msg: str,
) -> Tuple[bool, str]:
    """Uploads or updates a single file in GitHub via Contents REST API (zero `git` CLI required)."""
    repo_clean = repo.strip().strip("/")
    rel_clean = rel_path.replace("\\", "/").lstrip("/")
    api_url = f"https://api.github.com/repos/{repo_clean}/contents/{urllib.parse.quote(rel_clean, safe='/')}"
    headers = {
        "Authorization": f"Bearer {token.strip()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "TradingPro-v3-Sync",
    }

    existing_sha: Optional[str] = None
    existing_b64: Optional[str] = None
    get_url = f"{api_url}?ref={urllib.parse.quote(branch.strip() or 'main')}"
    req_get = urllib.request.Request(get_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req_get, timeout=20) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
            existing_sha = meta.get("sha")
            existing_b64 = meta.get("content", "")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            detail = exc.read().decode("utf-8", errors="ignore")
            return False, f"GitHub GET {exc.code}: {detail[:180]}"
    except Exception as exc:
        return False, f"Verbindungsfehler: {exc}"

    new_b64 = base64.b64encode(content_bytes).decode("ascii")
    if existing_b64 and "".join(existing_b64.split()) == new_b64:
        return True, "unverändert"

    payload: Dict[str, Any] = {
        "message": commit_msg,
        "content": new_b64,
        "branch": branch.strip() or "main",
    }
    if existing_sha:
        payload["sha"] = existing_sha

    body = json.dumps(payload).encode("utf-8")
    put_headers = dict(headers)
    put_headers["Content-Type"] = "application/json"
    req_put = urllib.request.Request(api_url, data=body, headers=put_headers, method="PUT")
    try:
        with urllib.request.urlopen(req_put, timeout=25) as resp:
            if resp.status in (200, 201):
                return True, "aktualisiert" if existing_sha else "erstellt"
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return False, f"GitHub PUT {exc.code}: {detail[:200]}"
    except Exception as exc:
        return False, f"Upload-Fehler: {exc}"


def push_watchlist_to_github(only_if_enabled: bool = False) -> Tuple[bool, str]:
    """Exports local DB -> watchlist.json and pushes it to GitHub via REST API."""
    cfg = load_local_cloud_config()
    if only_if_enabled and not bool(cfg.get("auto_sync_github", True)):
        return False, "Auto-Sync deaktiviert"

    repo = str(cfg.get("github_repo", "")).strip()
    token = str(cfg.get("github_token", "")).strip()
    branch = str(cfg.get("github_branch", "main")).strip() or "main"
    if not repo or not token:
        return False, "GitHub Repo oder Token nicht konfiguriert"

    synced = sync_watchlist_from_local_db()
    if not os.path.exists(WATCHLIST_FILE):
        return False, "watchlist.json konnte nicht erzeugt werden"

    with open(WATCHLIST_FILE, "rb") as fh:
        raw_bytes = fh.read()

    # Ignore synced_at timestamp differences by checking semantic payload hash against cached hash
    n_t = len(synced["tickers"]) if synced else 0
    n_d = len(synced["depot"]) if synced else 0
    ok, info = _github_put_file(
        repo=repo,
        branch=branch,
        token=token,
        rel_path="watchlist.json",
        content_bytes=raw_bytes,
        commit_msg=f"chore: sync watchlist ({n_t} tickers, {n_d} depot)",
    )
    if not ok:
        return False, info
    if info == "unverändert":
        return True, f"☁️ Cloud bereits aktuell ({n_t} Aktien)"
    return True, f"☁️ Cloud-Sync OK ({n_t} Aktien, {n_d} Depot)"


CLOUD_UPLOAD_FILES: List[str] = [
    ".gitignore",
    "requirements_monitor.txt",
    "run_monitor.py",
    "watchlist.json",
    "trading_pro/__init__.py",
    "trading_pro/data_loader.py",
    "trading_pro/macro.py",
    "trading_pro/strategy.py",
    "trading_pro/telegram_monitor.py",
    "docs/index.html",
    "docs/manifest.json",
    ".github/workflows/daily_monitor.yml",
]


def upload_project_to_github(
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[bool, str]:
    """One-click bootstrap upload of all cloud-monitor files into a GitHub repo (zero `git` needed)."""
    cfg = load_local_cloud_config()
    repo = str(cfg.get("github_repo", "")).strip()
    token = str(cfg.get("github_token", "")).strip()
    branch = str(cfg.get("github_branch", "main")).strip() or "main"
    if not repo or not token:
        return False, "Bitte zuerst GitHub Repo (owner/repo) und Token eintragen."

    sync_watchlist_from_local_db()
    total = len(CLOUD_UPLOAD_FILES)
    uploaded = 0
    skipped = 0
    errors: List[str] = []

    for idx, rel in enumerate(CLOUD_UPLOAD_FILES, 1):
        if on_progress:
            on_progress(idx, total, rel)
        abs_p = os.path.join(PROJECT_ROOT, rel.replace("/", os.sep))
        if not os.path.exists(abs_p):
            skipped += 1
            continue
        with open(abs_p, "rb") as fh:
            data = fh.read()
        ok, info = _github_put_file(
            repo=repo,
            branch=branch,
            token=token,
            rel_path=rel,
            content_bytes=data,
            commit_msg=f"chore: upload {rel}",
        )
        if ok:
            uploaded += 1
        else:
            if rel.startswith(".github/workflows/"):
                errors.append(f"{rel} (Token braucht Recht 'Workflows: Read & Write' — {info})")
            else:
                errors.append(f"{rel}: {info}")

    if errors:
        return False, f"{uploaded}/{total} hochgeladen. Fehler:\n" + "\n".join(errors[:4])
    return True, f"Alle {uploaded} Dateien erfolgreich ins GitHub-Repo '{repo}' hochgeladen!"


def send_telegram_message(
    token: str,
    chat_id: str,
    text_html: str,
    disable_preview: bool = True,
) -> Tuple[bool, str]:
    """Sends HTML-formatted message(s) via Telegram Bot API using Python stdlib only."""
    if not token or not chat_id:
        return False, "TELEGRAM_BOT_TOKEN oder TELEGRAM_CHAT_ID fehlt."

    # Telegram limit is 4096 chars per message -> split cleanly on line boundaries
    chunks: List[str] = []
    cur = ""
    for line in text_html.splitlines(keepends=True):
        if len(cur) + len(line) > 3900:
            chunks.append(cur.rstrip())
            cur = line
        else:
            cur += line
    if cur.strip():
        chunks.append(cur.rstrip())

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chunk in chunks:
        body = urllib.parse.urlencode(
            {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": "true" if disable_preview else "false",
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status != 200:
                    return False, f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            return False, f"HTTP {exc.code}: {detail[:200]}"
        except Exception as exc:
            return False, str(exc)
    return True, f"{len(chunks)} Nachricht(en) gesendet"


def evaluate_all_signals(
    wl_cfg: Dict[str, Any],
    use_cache_only: bool = False,
) -> Dict[str, Any]:
    """Downloads/loads macro + all watchlist tickers and computes Core R1-F, Macro & Supertrend signals."""
    settings = wl_cfg["settings"]
    tickers_meta = wl_cfg["tickers"]
    depot_set = {str(t).upper() for t in wl_cfg.get("depot", [])}

    macro_df = load_macro_data(force_download=not use_cache_only)
    core = get_core_allocation_status(macro_df)
    snap = get_latest_macro_snapshot(macro_df)
    diag = get_macro_flip_diagnostics(macro_df)

    ticker_list = [str(item["ticker"]) for item in tickers_meta]
    name_map = {str(item["ticker"]): str(item.get("name") or item["ticker"]) for item in tickers_meta}

    if not use_cache_only and ticker_list:
        download_tickers_parallel(ticker_list, force_download=True)

    rows: List[Dict[str, Any]] = []
    failed = 0
    for sym in ticker_list:
        df_px, _msg, _t = load_ticker_data(sym, "yahoo", force_download=False)
        if df_px is None or len(df_px) < 40:
            failed += 1
            continue
        res = run_strategy(
            df_px,
            ticker=sym,
            comp_name=name_map.get(sym, sym),
            config=settings,
            macro_df=macro_df,
        )
        if res is None:
            failed += 1
            continue
        sizing = calculate_position_size(
            portfolio_eur=float(settings.get("portfolio_eur", 25000.0)),
            risk_pct=float(settings.get("risk_per_trade_pct", 1.0)),
            max_pos_pct=float(settings.get("max_position_pct", 15.0)),
            entry_price=float(res.last_close),
            stop_price=float(res.stop_price),
        )
        rows.append(
            {
                "ticker": sym,
                "name": name_map.get(sym, sym),
                "in_depot": sym.upper() in depot_set,
                "signal": res.current_signal,
                "daily_regime": int(res.daily_regime),
                "signal_date": res.signal_date,
                "last_close": round(float(res.last_close), 4),
                "stop_price": round(float(res.stop_price), 4),
                "stop_dist_pct": round(float(res.stop_dist_pct), 2),
                "ext50_pct": round(float(res.ext50_pct), 2),
                "mom_score": round(float(res.mom_score), 2),
                "rsi_14": round(float(res.rsi_14), 1),
                "profit_factor": round(float(res.profit_factor), 2),
                "pos_eur": round(float(sizing.get("alloc_eur", 0.0)), 0),
                "shares": int(sizing.get("shares", 0)),
            }
        )

    total = len(rows)
    daily_up = sum(1 for r in rows if r["daily_regime"] == 1)
    n_buy = sum(1 for r in rows if r["signal"] == "BUY")
    n_wait = sum(1 for r in rows if r["signal"] == "WAIT")
    n_ext = sum(1 for r in rows if r["signal"] == "EXTENDED")
    breadth_pct = (daily_up / total * 100.0) if total > 0 else 0.0

    sig_rank = {"BUY": 0, "WAIT": 1, "EXTENDED": 2, "SELL": 3}
    rows.sort(key=lambda r: (sig_rank.get(r["signal"], 9), -r["mom_score"]))

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC"),
        "as_of": str(core.get("as_of") or snap.get("as_of") or "—"),
        "core": core,
        "macro_snap": snap,
        "macro_diag": diag,
        "breadth": {
            "total": total,
            "failed": failed,
            "daily_up": daily_up,
            "pct": round(breadth_pct, 1),
            "buy": n_buy,
            "wait": n_wait,
            "extended": n_ext,
        },
        "rows": rows,
        "settings": settings,
    }


def _stage_action_hint(old_stage: Optional[int], new_stage: int) -> str:
    if new_stage == 0:
        return "→ Depot A &amp; B verkaufen · 100 % DBX0AN"
    if new_stage == 1:
        if old_stage == 0 or old_stage is None:
            return "→ 50 % A0LC12 in Depot A kaufen · 50 % DBX0AN"
        return "→ 50 % A0LC12 (Depot A) · 50 % DBX0AN halten"
    if new_stage == 2:
        if old_stage == 3:
            return "→ Nur Depot B (50 % A0LC12) verkaufen → DBX0AN"
        if old_stage == 1:
            return "→ Bleibt 50 % A0LC12 / 50 % DBX0AN (kein Trade)"
        return "→ 50 % A0LC12 in Depot A kaufen · 50 % DBX0AN"
    return "→ DBX0AN (Depot B) in A0LC12 umschichten (100 % · 2×)"


def _short_core_next_step(core: Dict[str, Any]) -> str:
    st = int(core["stage"])
    qqq = float(core["qqq"])
    if st == 0:
        sma50 = float(core["sma50"])
        need = ((sma50 / qqq) - 1.0) * 100.0 if qqq > 0 else 0.0
        if need <= 0:
            return f"Einstieg 50 %: QQQ über 50er ({sma50:.0f}) — wartet auf Bestätigung"
        return f"Einstieg 50 % ab QQQ &gt; {sma50:.0f} ({need:+.1f} %)"
    if st == 1:
        up = float(core["sma200_up"])
        need = ((up / qqq) - 1.0) * 100.0 if qqq > 0 else 0.0
        return f"Bullenmarkt ab QQQ &gt; {up:.0f} ({need:+.1f} %) · Exit &lt; {core['sma50']:.0f}"
    if st == 2:
        wait_for = []
        if not core["smart_fed_ok"]:
            wait_for.append(f"Zins ≤4 %/GRÜN ({core['irx']:.2f} %)")
        if not core["vol_ok"]:
            wait_for.append(f"Vola &lt;20 % ({core['vol20']:.0f} %)")
        if not core["overheat_ok"]:
            wait_for.append("Abkühlung &lt;+22 %")
        w_txt = " + ".join(wait_for) if wait_for else "Freitag-Check"
        return f"2×-Turbo ab: {w_txt} · Exit &lt; {core['sma200_dn']:.0f}"
    return f"Bremse auf 50 % bei Vola &gt;28 % ({core['vol20']:.0f} %) oder Zins &gt;4 %↑ · Exit &lt; {core['sma200_dn']:.0f}"


def build_alerts_and_messages(
    report: Dict[str, Any],
    prev_state: Dict[str, Any],
    force_summary: bool = False,
    dash_url: str = "",
) -> Tuple[List[str], Dict[str, Any]]:
    """Computes diff vs. `prev_state` and builds ultra-clean, scannable Telegram alerts."""
    core = report["core"]
    snap = report["macro_snap"]
    breadth = report["breadth"]
    rows = report["rows"]
    as_of = report["as_of"]

    is_first_run = not bool(prev_state)
    prev_signals: Dict[str, str] = dict(prev_state.get("signals") or {})
    curr_signals: Dict[str, str] = {r["ticker"]: r["signal"] for r in rows}

    messages: List[str] = []

    # --- 1. URGENT ACTION ALERT (only sent when something actually changed!) ---
    urgent_blocks: List[str] = []

    if core.get("available"):
        prev_core_stage = prev_state.get("core_stage")
        prev_live_stage = prev_state.get("core_live_stage")
        curr_stage = int(core["stage"])
        curr_live = int(core["live_stage"])

        if prev_core_stage is not None and curr_stage != int(prev_core_stage):
            c_em = {0: "🔴", 1: "🟠", 2: "🟡", 3: "🟢"}.get(curr_stage, "⚪")
            urgent_blocks.append(
                f"🏦 <b>CORE R1-F: {c_em} Stufe {prev_core_stage} → Stufe {curr_stage}</b>\n"
                f"{_stage_action_hint(int(prev_core_stage), curr_stage)}"
            )
        elif (
            core.get("weekly_pending")
            and prev_live_stage is not None
            and curr_live != int(prev_live_stage)
        ):
            urgent_blocks.append(
                f"⏳ <b>CORE VORSCHAU:</b> Live Stufe {curr_live} (aktiv: Stufe {curr_stage})\n"
                f"→ Erst am Freitagabend umschichten"
            )

    if snap.get("available"):
        prev_regime = prev_state.get("macro_regime")
        curr_regime = str(snap["regime_label"])
        prev_fed = prev_state.get("fed_ok")
        curr_fed = bool(snap["fed_ok"])
        if prev_regime is not None and curr_regime != prev_regime:
            m_em = "🟢" if snap.get("fed_qqq50_ok") else ("🟡" if (curr_fed or snap.get("qqq50_ok")) else "🔴")
            urgent_blocks.append(
                f"🌍 <b>MAKRO: {m_em} {html.escape(curr_regime)}</b> (war {html.escape(str(prev_regime))})"
            )
        elif prev_fed is not None and curr_fed != bool(prev_fed):
            urgent_blocks.append(
                f"🌍 <b>FED 3M:</b> {'🟢 jetzt GRÜN' if curr_fed else '🔴 jetzt ROT'} ({snap['irx']:.2f} %)"
            )

    depot_sells = [
        r for r in rows
        if r["in_depot"] and r["signal"] == "SELL" and prev_signals.get(r["ticker"]) not in (None, "SELL")
    ]
    depot_buys = [
        r for r in rows
        if r["in_depot"] and r["signal"] == "BUY" and prev_signals.get(r["ticker"]) not in (None, "BUY")
    ]
    depot_near_stop = [
        r for r in rows
        if r["in_depot"] and r["daily_regime"] == 1 and 0.0 <= r["stop_dist_pct"] <= 3.0
    ]
    if depot_sells:
        lines = ["🔴 <b>DEPOT VERKAUFEN (SELL):</b>"]
        for r in depot_sells:
            lines.append(f"• <b>{html.escape(r['ticker'])}</b> · {r['last_close']:.2f} (Stop gerissen)")
        urgent_blocks.append("\n".join(lines))
    if depot_near_stop:
        lines = ["⚠️ <b>DEPOT NAHE STOP (&lt;3 %):</b>"]
        for r in depot_near_stop:
            lines.append(f"• <b>{html.escape(r['ticker'])}</b> · {r['last_close']:.2f} (Stop {r['stop_price']:.2f})")
        urgent_blocks.append("\n".join(lines))
    if depot_buys:
        lines = ["🟢 <b>DEPOT WIEDER BUY:</b>"]
        for r in depot_buys:
            lines.append(f"• <b>{html.escape(r['ticker'])}</b> · {r['last_close']:.2f} (Stop {r['stop_price']:.2f})")
        urgent_blocks.append("\n".join(lines))

    new_buys = [
        r for r in rows
        if r["signal"] == "BUY" and prev_signals.get(r["ticker"]) not in (None, "BUY")
    ]
    if new_buys:
        lines = [f"🚀 <b>NEUE KAUFSIGNALE ({len(new_buys)}):</b>"]
        for r in new_buys[:8]:
            pos_txt = f" · ~{r['pos_eur']:.0f} €" if r["pos_eur"] > 0 else ""
            lines.append(
                f"• <b>{html.escape(r['ticker'])}</b> · {r['last_close']:.2f} (Stop {r['stop_price']:.2f}{pos_txt})"
            )
        if len(new_buys) > 8:
            lines.append(f"<i>+ {len(new_buys) - 8} weitere im Dashboard</i>")
        urgent_blocks.append("\n".join(lines))

    if urgent_blocks:
        footer = f'\n\n📱 <a href="{html.escape(dash_url)}">Details im Dashboard</a>' if dash_url else ""
        messages.append(
            f"🚨 <b>HANDLUNGSBEDARF · {html.escape(as_of)}</b>\n\n"
            + "\n\n".join(urgent_blocks)
            + footer
        )

    # --- 2. ULTRA-CLEAN WEEKLY FRIDAY SUMMARY (or --force-summary) ---
    is_friday = bool(core.get("is_friday", False))
    last_summary_date = str(prev_state.get("last_summary_date", ""))
    need_summary = force_summary or is_first_run or (is_friday and last_summary_date != as_of)

    if need_summary:
        blocks = [f"📊 <b>TradingPro · {html.escape(as_of)}</b>"]

        if core.get("available"):
            c_em = {0: "🔴", 1: "🟠", 2: "🟡", 3: "🟢"}.get(int(core["stage"]), "⚪")
            short_names = {0: "Cash", 1: "V-Frühtranche", 2: "Bullenmarkt 1×", 3: "Voller Turbo 2×"}
            st_lbl = short_names.get(int(core["stage"]), str(core["stage_name"]))
            blocks.append(
                f"🏦 <b>CORE (70 %):</b> {c_em} Stufe {core['stage']} · {st_lbl}\n"
                f"→ <b>{core['a0lc12_pct']} % A0LC12 · {core['dbx0an_pct']} % DBX0AN</b>\n"
                f"→ {_short_core_next_step(core)}"
            )

        if snap.get("available"):
            if snap.get("fed_qqq50_ok"):
                m_txt = "🟢 GRÜN (Käufe erlaubt)"
            elif snap.get("fed_ok") or snap.get("qqq50_ok"):
                m_txt = "🟡 ÜBERGANG (Abwarten)"
            else:
                m_txt = "🔴 ROT (Abwarten)"
            blocks.append(
                f"🌍 <b>MAKRO (30 %):</b> {m_txt}\n"
                f"→ Breite: {breadth['pct']:.1f} % ({breadth['buy']} BUY · {breadth['wait']} WAIT)"
            )

        depot_rows = [r for r in rows if r["in_depot"]]
        if depot_rows:
            d_lines = ["💼 <b>MEIN DEPOT:</b>"]
            for r in depot_rows:
                em = {"BUY": "🟢", "WAIT": "🟡", "EXTENDED": "🟣", "SELL": "🔴"}.get(r["signal"], "⚪")
                d_lines.append(f"{em} <b>{html.escape(r['ticker'])}</b> ({r['signal']}) · {r['last_close']:.2f} · Stop {r['stop_dist_pct']:+.1f} %")
            blocks.append("\n".join(d_lines))

        buys_now = [r for r in rows if r["signal"] == "BUY"]
        if buys_now:
            b_lines = [f"🚀 <b>AKTIVE BUY-SIGNALE ({len(buys_now)}):</b>"]
            for r in buys_now[:5]:
                b_lines.append(f"• <b>{html.escape(r['ticker'])}</b> · {r['last_close']:.2f} (Stop {r['stop_price']:.2f})")
            blocks.append("\n".join(b_lines))
        else:
            blocks.append("✅ <b>Kein Handlungsbedarf.</b>")

        if dash_url:
            blocks.append(f'📱 <a href="{html.escape(dash_url)}">Alle Details im Dashboard</a>')

        messages.append("\n\n".join(blocks))

    new_state = {
        "updated_at": report["generated_at"],
        "as_of": as_of,
        "core_stage": int(core["stage"]) if core.get("available") else None,
        "core_live_stage": int(core["live_stage"]) if core.get("available") else None,
        "macro_regime": str(snap["regime_label"]) if snap.get("available") else None,
        "fed_ok": bool(snap["fed_ok"]) if snap.get("available") else None,
        "last_summary_date": as_of if need_summary else last_summary_date,
        "signals": curr_signals,
    }
    return messages, new_state


def generate_mobile_dashboard(report: Dict[str, Any], out_dir: str = DOCS_DIR) -> str:
    """Generates a self-contained dark-mode Mobile PWA (`docs/index.html` + `manifest.json`)."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "name": "TradingPro Mobile",
        "short_name": "TradingPro",
        "start_url": "./index.html",
        "display": "standalone",
        "background_color": "#12151a",
        "theme_color": "#12151a",
        "description": "Core R1-F Ampel, Makro-Liquidität & 128-Aktien Supertrend Monitor",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    # Embed sanitized JSON payload for fast client-side filtering & search
    data_payload = {
        "generated_at": report["generated_at"],
        "as_of": report["as_of"],
        "core": {k: v for k, v in report["core"].items() if not isinstance(v, pd.Series)},
        "macro_snap": report["macro_snap"],
        "macro_diag": {k: v for k, v in report["macro_diag"].items() if not isinstance(v, pd.Series)},
        "breadth": report["breadth"],
        "rows": report["rows"],
    }
    json_blob = json.dumps(data_payload, ensure_ascii=False, default=str).replace("</", "<\\/")

    html_doc = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <meta name="theme-color" content="#12151a" />
  <meta name="apple-mobile-web-app-capable" content="yes" />
  <link rel="manifest" href="manifest.json" />
  <title>TradingPro Mobile — Core R1-F &amp; Signale</title>
  <style>
    :root {{
      --bg: #12151a; --card: #1b2028; --card2: #222834; --border: #2e3646;
      --text: #eef2f7; --muted: #9aa6b8;
      --green: #2ecc71; --yellow: #f1c40f; --orange: #ff9f43; --red: #ff4d4d; --blue: #0dcaf0; --purple: #b388ff;
    }}
    * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); padding-bottom: 72px; }}
    header {{ position: sticky; top: 0; z-index: 20; background: rgba(18,21,26,0.95); backdrop-filter: blur(8px); border-bottom: 1px solid var(--border); padding: 12px 14px; display: flex; justify-content: space-between; align-items: center; }}
    header h1 {{ margin: 0; font-size: 16px; font-weight: 700; letter-spacing: 0.2px; }}
    header .meta {{ font-size: 11px; color: var(--muted); }}
    main {{ max-width: 780px; margin: 0 auto; padding: 12px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 14px; margin-bottom: 12px; }}
    .card-title {{ font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted); margin: 0 0 10px 0; }}
    .stage-banner {{ border-radius: 10px; padding: 12px 14px; font-weight: 700; font-size: 15px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px; }}
    .bg-red {{ background: rgba(255,77,77,0.18); border: 1px solid var(--red); color: #ffd4d4; }}
    .bg-orange {{ background: rgba(255,159,67,0.18); border: 1px solid var(--orange); color: #ffe6cc; }}
    .bg-blue {{ background: rgba(13,202,240,0.16); border: 1px solid var(--blue); color: #d1f4ff; }}
    .bg-green {{ background: rgba(46,204,113,0.18); border: 1px solid var(--green); color: #d5f7e2; }}
    .bg-yellow {{ background: rgba(241,196,15,0.18); border: 1px solid var(--yellow); color: #fff4c2; }}
    .alloc-bar {{ height: 10px; border-radius: 99px; background: #2c3442; overflow: hidden; margin: 10px 0 8px 0; display: flex; }}
    .alloc-a0 {{ background: linear-gradient(90deg, var(--green), var(--blue)); height: 100%; }}
    .lights {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(125px, 1fr)); gap: 8px; margin-top: 10px; }}
    .light-pill {{ background: var(--card2); border: 1px solid var(--border); border-radius: 10px; padding: 8px 10px; display: flex; align-items: center; gap: 8px; font-size: 12px; }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
    .dot-green {{ background: var(--green); box-shadow: 0 0 8px rgba(46,204,113,0.7); }}
    .dot-yellow {{ background: var(--yellow); box-shadow: 0 0 8px rgba(241,196,15,0.7); }}
    .dot-red {{ background: var(--red); box-shadow: 0 0 8px rgba(255,77,77,0.7); }}
    .trigger {{ margin-top: 10px; font-size: 12px; color: #cbd5e1; line-height: 1.45; background: rgba(255,255,255,0.03); padding: 9px 10px; border-radius: 8px; border-left: 3px solid var(--blue); }}
    .macro-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }}
    .m-box {{ background: var(--card2); border: 1px solid var(--border); border-radius: 10px; padding: 10px; font-size: 12px; }}
    .m-box .lbl {{ color: var(--muted); font-size: 11px; margin-bottom: 4px; display: flex; align-items: center; gap: 6px; }}
    .m-box .val {{ font-weight: 700; font-size: 14px; }}
    .m-box .sub {{ color: var(--muted); font-size: 11px; margin-top: 2px; }}
    .controls {{ display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }}
    input[type="search"] {{ flex: 1; min-width: 150px; background: var(--card2); color: var(--text); border: 1px solid var(--border); border-radius: 10px; padding: 9px 12px; font-size: 14px; outline: none; }}
    .chip {{ background: var(--card2); color: var(--muted); border: 1px solid var(--border); border-radius: 99px; padding: 6px 12px; font-size: 12px; font-weight: 600; cursor: pointer; }}
    .chip.active {{ background: var(--blue); color: #0a141c; border-color: var(--blue); }}
    .stock-row {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 4px; border-bottom: 1px solid rgba(46,54,70,0.6); gap: 8px; }}
    .stock-row:last-child {{ border-bottom: none; }}
    .st-left {{ min-width: 0; }}
    .st-ticker {{ font-weight: 700; font-size: 14px; display: flex; align-items: center; gap: 6px; }}
    .st-name {{ font-size: 11px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 185px; }}
    .st-right {{ text-align: right; font-size: 12px; flex-shrink: 0; }}
    .badge {{ display: inline-block; padding: 2px 7px; border-radius: 6px; font-size: 10px; font-weight: 700; }}
    .b-BUY {{ background: rgba(46,204,113,0.2); color: var(--green); border: 1px solid var(--green); }}
    .b-WAIT {{ background: rgba(241,196,15,0.18); color: var(--yellow); border: 1px solid var(--yellow); }}
    .b-EXTENDED {{ background: rgba(179,136,255,0.2); color: var(--purple); border: 1px solid var(--purple); }}
    .b-SELL {{ background: rgba(255,77,77,0.18); color: var(--red); border: 1px solid var(--red); }}
    .nav {{ position: fixed; bottom: 0; left: 0; right: 0; background: #161a21; border-top: 1px solid var(--border); display: flex; z-index: 30; padding-bottom: env(safe-area-inset-bottom); }}
    .nav button {{ flex: 1; background: none; border: none; color: var(--muted); padding: 10px 4px; font-size: 12px; font-weight: 600; cursor: pointer; }}
    .nav button.active {{ color: var(--blue); border-top: 2px solid var(--blue); }}
    .tab {{ display: none; }}
    .tab.active {{ display: block; }}
    .empty {{ text-align: center; color: var(--muted); padding: 26px 12px; font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>TradingPro Mobile</h1>
      <div class="meta" id="hdr-meta">Stand: —</div>
    </div>
    <div id="hdr-breadth" class="badge b-SELL">—</div>
  </header>

  <main>
    <section id="tab-core" class="tab active">
      <div class="card" id="core-card"></div>
      <div class="card" id="macro-card"></div>
    </section>

    <section id="tab-depot" class="tab">
      <div class="card">
        <div class="card-title">💼 Mein Depot</div>
        <div id="depot-list"></div>
      </div>
    </section>

    <section id="tab-watchlist" class="tab">
      <div class="card">
        <div class="card-title" id="wl-title">📋 Watchlist</div>
        <div class="controls">
          <input type="search" id="search-box" placeholder="Ticker oder Name suchen..." />
        </div>
        <div class="controls" id="filter-chips"></div>
        <div id="wl-list"></div>
      </div>
    </section>
  </main>

  <nav class="nav">
    <button class="active" data-tab="tab-core">🌍 Core &amp; Makro</button>
    <button data-tab="tab-depot">💼 Depot</button>
    <button data-tab="tab-watchlist">📋 Watchlist</button>
  </nav>

  <script>
    const D = {json_blob};
    const fmt = (n, d=2) => (n === null || n === undefined || isNaN(n)) ? "—" : Number(n).toFixed(d);
    const fmtS = (n, d=1) => (n === null || n === undefined || isNaN(n)) ? "—" : (Number(n) >= 0 ? "+" : "") + Number(n).toFixed(d);

    document.getElementById("hdr-meta").textContent = "Stand: " + D.as_of + " (" + D.generated_at + ")";
    const br = D.breadth;
    const hb = document.getElementById("hdr-breadth");
    hb.textContent = "Breite " + fmt(br.pct,1) + "% (" + br.buy + " BUY)";
    hb.className = "badge " + (br.pct >= 35 ? "b-BUY" : (br.pct >= 20 ? "b-WAIT" : "b-SELL"));

    // Render Core R1-F Card
    const c = D.core;
    const stageClass = {{0:"bg-red", 1:"bg-orange", 2:"bg-blue", 3:"bg-green"}}[c.stage] || "bg-red";
    const volDot = (c.vol_ok && c.vol20 < 20) ? "dot-green" : (c.vol_stress ? "dot-red" : (c.vol_ok ? "dot-yellow" : "dot-red"));
    const ohDot = (c.overheat_ok && c.vs_sma200_pct < 22) ? "dot-green" : (!c.overheat_ok ? "dot-red" : "dot-yellow");
    document.getElementById("core-card").innerHTML = c.available ? `
      <div class="card-title">🏦 70%-Core R1-F (A0LC12 / DBX0AN)</div>
      <div class="stage-banner ${{stageClass}}">
        <span>STUFE ${{c.stage}} — ${{c.stage_name}}</span>
        <span>${{c.a0lc12_pct}}% A0LC12 / ${{c.dbx0an_pct}}% DBX0AN (${{fmt(c.eff_lev,1)}}×)</span>
      </div>
      <div class="alloc-bar"><div class="alloc-a0" style="width:${{c.a0lc12_pct}}%"></div></div>
      <div class="lights">
        <div class="light-pill"><span class="dot ${{c.sma50_ok?'dot-green':'dot-red'}}"></span><div><b>50er</b><br>${{fmtS(c.vs_sma50_pct)}}%</div></div>
        <div class="light-pill"><span class="dot ${{c.sma200_band_ok?'dot-green':'dot-red'}}"></span><div><b>200er (+2,5/-3%)</b><br>${{fmtS(c.vs_sma200_pct)}}%</div></div>
        <div class="light-pill"><span class="dot ${{c.smart_fed_ok?'dot-green':'dot-red'}}"></span><div><b>Smart-Zins &gt;4%</b><br>${{fmt(c.irx)}}%</div></div>
        <div class="light-pill"><span class="dot ${{volDot}}"></span><div><b>Vola (20/28)</b><br>${{fmt(c.vol20,1)}}%</div></div>
        <div class="light-pill"><span class="dot ${{ohDot}}"></span><div><b>Überhitzung</b><br>${{c.overheat_ok ? 'OK (<30%)' : 'BREMS'}}</div></div>
      </div>
      <div class="trigger">➜ ${{c.next_trigger}}<br><span style="color:var(--muted);font-size:11px">QQQ: ${{fmt(c.qqq)}} | Freitags-Check | 2-Depot-LIFO | Drift-Cap 65%</span></div>
    ` : '<div class="empty">Keine Core-Daten verfügbar.</div>';

    // Render Macro Card
    const m = D.macro_snap;
    const dg = D.macro_diag;
    const mCls = m.fed_qqq50_ok ? "bg-green" : ((m.fed_ok || m.qqq50_ok) ? "bg-yellow" : "bg-red");
    const fedSub = dg.fed_ok
      ? `Puffer ${{fmt(Math.abs(dg.irx_diff_bp),0)}} Bp ≤ ${{fmt(dg.irx_th)}}%`
      : (dg.auto_flip_date ? `Auto-Flip ~${{dg.auto_flip_date}}` : `in 20T: ${{fmt(dg.irx_th_20d)}}% (${{fmtS(dg.irx_gap_20d_bp,0)}} Bp)`);
    document.getElementById("macro-card").innerHTML = m.available ? `
      <div class="card-title">🌍 Makro-Liquidität (30%-Satellit)</div>
      <div class="stage-banner ${{mCls}}" style="margin-bottom:10px">
        <span>${{m.regime_label}} (3T)</span>
        <span>Breite: ${{fmt(br.pct,1)}}% (${{br.daily_up}}/${{br.total}})</span>
      </div>
      <div class="macro-grid">
        <div class="m-box"><div class="lbl"><span class="dot ${{m.fed_ok?'dot-green':'dot-red'}}"></span>Fed 3M (^IRX)</div><div class="val">${{fmt(m.irx)}}%</div><div class="sub">${{fedSub}}</div></div>
        <div class="m-box"><div class="lbl"><span class="dot ${{m.tnx_ok?'dot-green':'dot-red'}}"></span>10J Rendite (^TNX)</div><div class="val">${{fmt(m.tnx)}}%</div><div class="sub">vs SMA50 ${{fmt(dg.tnx_ma)}}% (${{fmtS(dg.tnx_diff_bp,0)}} Bp)</div></div>
        <div class="m-box"><div class="lbl"><span class="dot ${{m.dxy_ok?'dot-green':'dot-red'}}"></span>US-Dollar (DXY)</div><div class="val">${{fmt(m.dxy,1)}}</div><div class="sub">vs SMA50 ${{fmt(dg.dxy_ma,1)}} (${{fmtS(dg.dxy_diff_pct)}}%)</div></div>
        <div class="m-box"><div class="lbl"><span class="dot ${{m.vix<22?'dot-green':(m.vix<=30?'dot-yellow':'dot-red')}}"></span>VIX &amp; Small-Caps</div><div class="val">VIX ${{fmt(m.vix,1)}}</div><div class="sub">IWM ${{fmtS(dg.iwm_diff_pct)}}% vs SMA50</div></div>
      </div>
    ` : '<div class="empty">Keine Makro-Daten verfügbar.</div>';

    // Stock Row Renderer
    function renderRows(containerId, list) {{
      const el = document.getElementById(containerId);
      if (!list.length) {{
        el.innerHTML = '<div class="empty">Keine Einträge in dieser Ansicht.</div>';
        return;
      }}
      el.innerHTML = list.map(r => `
        <div class="stock-row">
          <div class="st-left">
            <div class="st-ticker">${{r.ticker}} <span class="badge b-${{r.signal}}">${{r.signal}}</span></div>
            <div class="st-name">${{r.name}} · seit ${{r.signal_date}}</div>
          </div>
          <div class="st-right">
            <div><b>${{fmt(r.last_close)}}</b> · Stop ${{fmt(r.stop_price)}} (${{fmtS(r.stop_dist_pct)}}%)</div>
            <div style="color:var(--muted)">vs50: ${{fmtS(r.ext50_pct)}}% · Mom: ${{fmtS(r.mom_score)}} · ~${{Math.round(r.pos_eur)}} €</div>
          </div>
        </div>
      `).join("");
    }}

    // Depot Tab
    const depotRows = D.rows.filter(r => r.in_depot);
    renderRows("depot-list", depotRows);

    // Watchlist Tab with live search & filter chips
    let activeFilter = "ALL";
    document.getElementById("wl-title").textContent = `📋 Watchlist (${{D.rows.length}} Aktien)`;
    const counts = {{ALL: D.rows.length, BUY: br.buy, WAIT: br.wait, EXTENDED: br.extended, SELL: D.rows.length - br.buy - br.wait - br.extended}};
    const chipsEl = document.getElementById("filter-chips");
    [["ALL","Alle"],["BUY","BUY"],["WAIT","WAIT"],["EXTENDED","EXT"],["SELL","SELL"]].forEach(([k, lbl]) => {{
      const b = document.createElement("button");
      b.className = "chip" + (k === activeFilter ? " active" : "");
      b.textContent = `${{lbl}} (${{counts[k]}})`;
      b.onclick = () => {{
        activeFilter = k;
        chipsEl.querySelectorAll(".chip").forEach(x => x.classList.remove("active"));
        b.classList.add("active");
        applyWatchlistFilter();
      }};
      chipsEl.appendChild(b);
    }});

    function applyWatchlistFilter() {{
      const q = document.getElementById("search-box").value.trim().toUpperCase();
      const filtered = D.rows.filter(r => {{
        if (activeFilter !== "ALL" && r.signal !== activeFilter) return false;
        if (q && !r.ticker.toUpperCase().includes(q) && !r.name.toUpperCase().includes(q)) return false;
        return true;
      }});
      renderRows("wl-list", filtered);
    }}
    document.getElementById("search-box").addEventListener("input", applyWatchlistFilter);
    applyWatchlistFilter();

    // Bottom Navigation Tabs
    document.querySelectorAll(".nav button").forEach(btn => {{
      btn.onclick = () => {{
        document.querySelectorAll(".nav button").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(btn.dataset.tab).classList.add("active");
        window.scrollTo(0, 0);
      }};
    }});
  </script>
</body>
</html>
"""
    out_html = os.path.join(out_dir, "index.html")
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
    return out_html
