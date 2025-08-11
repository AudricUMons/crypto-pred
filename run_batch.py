#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, time, argparse, itertools
from datetime import datetime, timezone
import pandas as pd

# Imports adaptés à ta structure de dossiers
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
    os.makedirs(p, exist_ok=True); return p

def dict_hash(d):
    return str(sorted(d.items()))

def load_done_set(csv_path):
    if not os.path.exists(csv_path): return set()
    try:
        df = pd.read_csv(csv_path)
        if "param_hash" in df.columns:
            return set(df["param_hash"].astype(str).tolist())
    except Exception:
        pass
    return set()

def append_row(csv_path, row):
    import pandas as _pd
    df = _pd.DataFrame([row])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)

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
    args = parser.parse_args()

    # Charger la grille
    if args.param_grid_file:
        with open(args.param_grid_file, "r", encoding="utf-8") as f:
            param_grid = json.load(f)
    else:
        param_grid = user_config.PARAM_GRID

    # Préparer sorties
    results_dir = safe_mkdir(args.results_dir)
    live_csv = os.path.join(results_dir, "grid_results_live.csv")
    best_json = os.path.join(results_dir, "best_so_far.json")

    total = count_total_combos(param_grid)
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}] "
          f"RUN CONFIG → coin={args.coin_id}, vs={args.vs_currency}, provider={args.provider}, "
          f"results_dir={results_dir}, grid={total} combos")

    done = load_done_set(live_csv)

    # Reprise du meilleur en cours si présent
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

        # Données (avec cache)
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

        final_value = float(out_df["value"].iloc[-1])
        roi_pct = (final_value / args.initial_cash - 1.0) * 100.0
        row = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            **param_kv,
            "initial_cash": args.initial_cash,
            "final_value": round(final_value, 4),
            "roi_pct": round(roi_pct, 4),
            "trades_count": len(trades[0]),
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

        if args.cooldown_seconds > 0:
            time.sleep(args.cooldown_seconds)

if __name__ == "__main__":
    main()
