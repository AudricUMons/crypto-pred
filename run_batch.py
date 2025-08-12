#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, time, argparse, itertools, subprocess
from datetime import datetime, timezone
import pandas as pd

from crypto_app import config as user_config
from crypto_app.methods import data as user_data
from crypto_app.methods import features as user_features
from crypto_app.methods import models as user_models


def expand_param_grid(grid_blocks):
    for block in grid_blocks:
        keys = list(block.keys())
        for combo in itertools.product(*[block[k] for k in keys]):
            yield dict(zip(keys, combo))


def count_total_combos(grid_blocks):
    total = 0
    for block in grid_blocks:
        m = 1
        for v in block.values():
            m *= len(v)
        total += m
    return total


def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p


def dict_hash(d):
    return str(sorted(d.items()))


def load_done_set(csv_path):
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        if "param_hash" in df.columns:
            return set(df["param_hash"].astype(str).tolist())
    except Exception:
        pass
    return set()


def append_row(csv_path, row):
    df = pd.DataFrame([row])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)


def _compute_final_value(out_df, df_feat=None, df_raw=None):
    if out_df is None or len(out_df) == 0:
        return None
    if "value" in out_df.columns:
        return float(out_df["value"].iloc[-1])

    cash_last = float(out_df["cash"].iloc[-1]) if "cash" in out_df.columns else 0.0
    btc_last  = float(out_df["btc"].iloc[-1])  if "btc"  in out_df.columns else 0.0

    last_price = None
    for df in (df_feat, df_raw):
        if df is None:
            continue
        for col in ("price", "close", "Close", "adj_close", "Adj Close"):
            if col in df.columns:
                try:
                    last_price = float(df[col].iloc[-1])
                    break
                except Exception:
                    continue
        if last_price is not None:
            break

    if last_price is None:
        return None
    return cash_last + btc_last * last_price


# ---------- Git helpers ----------
def _run(cmd):
    subprocess.run(cmd, shell=True, check=False,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def git_checkpoint(message, results_dir, best_json):
    if not os.path.isdir(".git"):
        return
    branch = os.getenv("GITHUB_REF_NAME") or os.getenv("GITHUB_HEAD_REF") or "main"
    _run('git config user.name "github-actions[bot]"')
    _run('git config user.email "41898282+github-actions[bot]@users.noreply.github.com"')
    _run(f'git add {results_dir}/*.csv {results_dir}/*.json || true')
    if os.path.exists(best_json):
        _run(f'git add "{best_json}" || true')
    _run(f'git commit -m "{message} [skip ci]" || true')
    _run(f'git pull --rebase origin {branch} || true')
    _run('git push || true')
# ----------------------------------


def main():
    parser = argparse.ArgumentParser(description="Batch runner (sans UI) pour AUTO_TEST.")
    parser.add_argument("--coin-id", default=user_config.DEFAULT_COIN_ID)
    parser.add_argument("--vs-currency", default=user_config.DEFAULT_VS_CURRENCY)
    parser.add_argument("--provider", default="auto", help="binance|coincap|cryptocompare|auto")
    parser.add_argument("--results-dir", default=user_config.RESULTS_DIR)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--save-details", action="store_true",
                        default=getattr(user_config, "SAVE_PER_RUN_DETAILS", False))
    parser.add_argument("--param-grid-file", default=None,
                        help="JSON optionnel pour remplacer config.PARAM_GRID")
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--max-runtime-mins", type=float, default=0.0,
                        help="Stoppe proprement après N minutes (0=illimité)")
    parser.add_argument("--git-push-each", action="store_true",
                        help="Commit & push après CHAQUE combo (checkpoint).")
    parser.add_argument("--push-every", type=int, default=1,
                        help="Si --git-push-each, push toutes les N configs (défaut 1).")
    args = parser.parse_args()

    if args.param_grid_file:
        with open(args.param_grid_file, "r", encoding="utf-8") as f:
            param_grid = json.load(f)
    else:
        param_grid = user_config.PARAM_GRID

    results_dir = safe_mkdir(args.results_dir)
    live_csv = os.path.join(results_dir, "grid_results_live.csv")
    best_json = os.path.join(results_dir, "best_so_far.json")

    total = count_total_combos(param_grid)
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}] "
          f"RUN CONFIG → coin={args.coin_id}, vs={args.vs_currency}, provider={args.provider}, "
          f"results_dir={results_dir}, grid={total} combos")

    start_ts = time.time()
    deadline = start_ts + args.max_runtime_mins * 60 if args.max_runtime_mins and args.max_runtime_mins > 0 else None

    done = load_done_set(live_csv)
    print(f"Resuming: skipping {len(done)} already-done combos; remaining {total - len(done)} to run.")

    best_track = {"final_value": float("-inf"), "row": None}
    if os.path.exists(best_json):
        try:
            with open(best_json, "r", encoding="utf-8") as f:
                prev = json.load(f)
                best_track["final_value"] = float(prev.get("final_value", float("-inf")))
                best_track["row"] = prev.get("row")
        except Exception:
            pass

    data_cache = {}
    combos = list(expand_param_grid(param_grid))
    for i, params in enumerate(combos, start=1):
        if deadline and time.time() >= deadline:
            print("[STOP] Max runtime reached — exiting cleanly before starting next combo.")
            break

        days        = int(params.get("days", user_config.DEFAULT_DAYS))
        horizon     = int(params.get("horizon", getattr(user_config, "RF_HORIZON", 24)))
        n_lags      = int(params.get("n_lags", getattr(user_config, "RF_N_LAGS", 30)))
        train_step  = int(params.get("train_step", getattr(user_config, "RF_TRAIN_STEP", 1)))
        n_estimators= int(params.get("n_estimators", getattr(user_config, "RF_N_ESTIMATORS", 128)))
        fee_rate    = float(params.get("fee_rate", getattr(user_config, "RF_FEE_RATE", 0.001)))
        threshold   = float(params.get("threshold", getattr(user_config, "RF_THRESHOLD", 0.002)))

        param_kv = {
            "days": days, "horizon": horizon, "n_lags": n_lags, "train_step": train_step,
            "n_estimators": n_estimators, "fee_rate": fee_rate, "threshold": threshold,
            "coin_id": args.coin_id, "vs_currency": args.vs_currency, "provider": args.provider,
        }
        phash = dict_hash(param_kv)
        if phash in done:
            continue

        dkey = (days, args.coin_id, args.vs_currency, args.provider)
        if dkey not in data_cache:
            df_raw = user_data.fetch_data(days=days, coin_id=args.coin_id,
                                          vs_currency=args.vs_currency, provider=args.provider)
            data_cache[dkey] = df_raw
        else:
            df_raw = data_cache[dkey]

        df_feat = user_features.add_indicators(df_raw)

        out_df, trades = user_models.simulate_rf_follow(
            df_feat,
            initial_cash=args.initial_cash,
            horizon=horizon,
            n_lags=n_lags,
            train_step=train_step,
            n_estimators=n_estimators,
            fee_rate=fee_rate,
            threshold=threshold,
            return_trades=True,
            return_signals=False,
        )

        final_value = _compute_final_value(out_df, df_feat=df_feat, df_raw=df_raw)
        if final_value is None:
            print("[SKIP] résultat invalide (ni 'value' ni prix dispo) → combo ignoré.")
            continue

        roi_pct = (final_value / args.initial_cash - 1.0) * 100.0
        try:
            trades_count = len(trades[0]) if isinstance(trades, (list, tuple)) else int(trades)
        except Exception:
            trades_count = 0

        row = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            **param_kv,
            "initial_cash": args.initial_cash,
            "final_value": round(final_value, 4),
            "roi_pct": round(roi_pct, 4),
            "trades_count": trades_count,
            "param_hash": phash,
        }
        append_row(live_csv, row)

        if args.save_details:
            base = f"details_{args.coin_id}_{days}d_L{n_lags}_H{horizon}_T{train_step}_NE{n_estimators}_TH{threshold}_FEE{fee_rate}.csv"
            out_path = os.path.join(results_dir, base)
            out_df.to_csv(out_path)

        if final_value > best_track["final_value"]:
            best_track = {"final_value": final_value, "row": row}
            with open(best_json, "w", encoding="utf-8") as f:
                json.dump(best_track, f, ensure_ascii=False, indent=2)

        print(f"[{i}/{total}] days={days} L={n_lags} H={horizon} TS={train_step} "
              f"NE={n_estimators} TH={threshold} FEE={fee_rate} -> final=${final_value:,.2f} (ROI={roi_pct:.2f}%)")

        if args.git_push_each and (i % max(1, args.push_every) == 0):
            git_checkpoint(message=f"checkpoint {i}/{total}", results_dir=results_dir, best_json=best_json)

        if args.cooldown_seconds > 0:
            time.sleep(args.cooldown_seconds)


if __name__ == "__main__":
    main()
