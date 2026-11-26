"""CLI Entry Point for the Autonomous TradingPro Telegram Monitor & Mobile Dashboard.

Examples:
  python run_monitor.py --dry-run --use-cache --force-summary
  python run_monitor.py --sync-only
  python run_monitor.py --test-telegram
  python run_monitor.py --force-summary
"""

from __future__ import annotations

import argparse
import sys

from trading_pro.telegram_monitor import (
    build_alerts_and_messages,
    evaluate_all_signals,
    generate_mobile_dashboard,
    get_telegram_credentials,
    load_monitor_state,
    load_watchlist_config,
    save_monitor_state,
    send_telegram_message,
    sync_watchlist_from_local_db,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="TradingPro Autonomous Telegram & Mobile Dashboard Monitor")
    parser.add_argument("--dry-run", action="store_true", help="Kein Telegram-Versand und kein State-Update")
    parser.add_argument("--use-cache", action="store_true", help="Nur lokalen Kurs-Cache nutzen (schneller Offline-Test)")
    parser.add_argument("--force-summary", action="store_true", help="Immer vollständigen Status-Bericht erzeugen")
    parser.add_argument("--sync-only", action="store_true", help="Nur watchlist.json aus lokaler DB aktualisieren")
    parser.add_argument("--test-telegram", action="store_true", help="Kurze Telegram-Testnachricht senden")
    args = parser.parse_args()

    # Auto-sync watchlist.json if local desktop SQLite DB is present
    synced = sync_watchlist_from_local_db()
    if synced is not None:
        print(
            f"[Sync] watchlist.json aktualisiert: {len(synced['tickers'])} Aktien, "
            f"{len(synced['depot'])} im Depot."
        )
    if args.sync_only:
        return 0

    token, chat_id, dash_url = get_telegram_credentials()

    if args.test_telegram:
        ok, info = send_telegram_message(
            token,
            chat_id,
            "✅ <b>TradingPro Bot verbunden!</b>\nDu erhältst ab sofort zuverlässige Push-Alarme für Core R1-F, Makro-Ampel &amp; Watchlist.",
        )
        print(f"[Telegram Test] ok={ok} ({info})")
        return 0 if ok else 1

    wl_cfg = load_watchlist_config()
    prev_state = load_monitor_state()

    print(
        f"[Monitor] Prüfe Makro, Core R1-F & {len(wl_cfg['tickers'])} Aktien "
        f"({'Cache' if args.use_cache else 'Live-Download'})..."
    )
    report = evaluate_all_signals(wl_cfg, use_cache_only=args.use_cache)
    html_path = generate_mobile_dashboard(report)
    print(f"[Dashboard] Mobile PWA erstellt: {html_path}")

    messages, new_state = build_alerts_and_messages(
        report, prev_state, force_summary=args.force_summary, dash_url=dash_url
    )

    if not messages:
        print("[Monitor] Keine neuen Alarme / kein Handlungsbedarf heute.")
    else:
        for i, msg in enumerate(messages, 1):
            print(f"\n--- TELEGRAM NACHRICHT {i}/{len(messages)} ---\n{msg}\n")
            if not args.dry_run:
                ok, info = send_telegram_message(token, chat_id, msg)
                print(f"[Telegram] Versand {i}: ok={ok} ({info})")
                if not ok:
                    return 1

    if not args.dry_run:
        save_monitor_state(new_state)
        print("[State] monitor_state.json gespeichert.")
    else:
        print("[Dry-Run] Kein Versand & kein Überschreiben von monitor_state.json.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
