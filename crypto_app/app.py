import os
import io
import json
import math
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# --- ton app existante ---
from crypto_app.ui import setup_page
from crypto_app.sidebar import render_sidebar
from crypto_app.methods.data import fetch_data
from crypto_app.methods.features import add_indicators
from crypto_app.methods.models import simulate_rf_follow
from crypto_app.charts.main import render_main_chart
from crypto_app.charts.secondary import render_trades_chart

# --- config (AJOUT) ---
from . import config


# ---------- Helpers ----------
def _product_dict(d):
    # Si on reçoit une liste de sous-grilles, on les déroule
    if isinstance(d, list):
        for sub in d:
            yield from _product_dict(sub)
        return

    if not isinstance(d, dict):
        raise TypeError(f"PARAM_GRID doit être dict ou list[dict], pas {type(d)}")

    keys = list(d.keys())
    values = [v if isinstance(v, (list, tuple, set)) else [v] for v in (d[k] for k in keys)]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _call_simulate(df, params, want_signals=False):
    """
    Appelle simulate_rf_follow avec robustesse:
    - tente avec hyperparamètres étendus (n_estimators/threshold/fee_rate)
    - si non supporté, retombe sur la signature de base (n_lags/horizon/train_step)
    """
    kw_base = dict(
        n_lags=params["n_lags"],
        horizon=params["horizon"],
        train_step=params["train_step"],
        return_trades=True,
        return_signals=want_signals,
    )

    # Tente avec hyperparams supplémentaires si présents dans le grid / config
    kw_try = kw_base.copy()
    if "n_estimators" in params or hasattr(config, "RF_N_ESTIMATORS"):
        kw_try["n_estimators"] = params.get("n_estimators", getattr(config, "RF_N_ESTIMATORS", 100))
    if "threshold" in params or hasattr(config, "RF_THRESHOLD"):
        kw_try["threshold"] = params.get("threshold", getattr(config, "RF_THRESHOLD", 0.002))
    if "fee_rate" in params or hasattr(config, "RF_FEE_RATE"):
        kw_try["fee_rate"] = params.get("fee_rate", getattr(config, "RF_FEE_RATE", 0.001))

    try:
        return simulate_rf_follow(df, **kw_try)
    except TypeError:
        # Fallback sur la signature minimale
        return simulate_rf_follow(df, **kw_base)


def _evaluate_run(df, params):
    """
    Lance une simulation et calcule des métriques:
      - final_value, total_return, buy&hold, excess, max drawdown, trades_count
    Retourne: (pf, metrics, trades_tuple)
    """
    pf, (buys_d, buys_p, sells_d, sells_p) = _call_simulate(df, params, want_signals=False)

    pv0 = float(pf["portfolio_value"].iloc[0])
    pvN = float(pf["portfolio_value"].iloc[-1])
    total_return = (pvN / pv0) - 1.0

    bh_series = (df["price"] / df["price"].iloc[0]) * pv0
    bh_final = float(bh_series.iloc[-1])

    pv = pf["portfolio_value"].to_numpy(dtype=float)
    running_max = np.maximum.accumulate(pv)
    dd = (pv - running_max) / running_max
    max_dd = float(dd.min()) if len(dd) else 0.0

    metrics = {
        "final_value": pvN,
        "total_return_%": 100 * total_return,
        "bh_final_value": bh_final,
        "excess_vs_bh": pvN - bh_final,
        "max_drawdown_%": 100 * max_dd,  # négatif ou 0
        "trades_count": len(buys_d) + len(sells_d),
    }
    return pf, metrics, (buys_d, buys_p, sells_d, sells_p)


def _save_results(res_df, best_tuple, df, out_dir):
    """
    Sauvegarde CSV global + meta (+ bouton de DL).
    res_df: DataFrame de toutes les configs triées
    best_tuple: (score, params, pf, trades)
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "grid_results.csv")
    res_df.to_csv(csv_path, index=False)

    score, best_params, best_pf, _ = best_tuple
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data": {
            "days": getattr(config, "DEFAULT_DAYS", None),
            "coin_id": getattr(config, "DEFAULT_COIN_ID", None),
            "vs_currency": getattr(config, "DEFAULT_VS_CURRENCY", None),
            "rows": int(df.shape[0]),
        },
        "grid": getattr(config, "PARAM_GRID", {}),
        "best": {"score_final_value": float(score), "params": best_params},
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(csv_path, "rb") as f:
        st.download_button(
            "⬇️ Télécharger toutes les configurations (CSV)",
            f,
            file_name=os.path.basename(csv_path),
            mime="text/csv",
        )

    st.caption(f"Résultats sauvegardés dans: `{out_dir}`")
    return csv_path


# ---------- App ----------
def main():
    setup_page()

    # ------- MODE AUTO TEST -------
    if getattr(config, "AUTO_TEST", False):
        st.header("🔁 Auto-test des paramètres (config.PARAM_GRID)")

        combos = list(_product_dict(getattr(config, "PARAM_GRID", {})))
        if not combos:
            st.error("PARAM_GRID est vide dans config.py — rien à tester.")
            return

        # Dossier de sortie + fichier CSV incrémental
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        root = getattr(config, "RESULTS_DIR", "results")
        out_dir = os.path.join(root, f"autotest_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        csv_live = os.path.join(out_dir, "grid_results_live.csv")

        st.write(f"Nombre de combinaisons à tester : **{len(combos)}**")
        progress = st.progress(0)

        results = []
        best = None            # (score, params, pf, trades)
        best_days = None       # pour recharger le bon df pour les graphes/sauvegardes
        prev_days = None
        df = None

        for i, params in enumerate(combos, start=1):
            # 1) nombre de jours pour cette configuration (sinon DEFAULT_DAYS)
            days = int(params.get("days", getattr(config, "DEFAULT_DAYS", 90)))

            # 2) (re)chargement si 'days' a changé
            if df is None or days != prev_days:
                with st.spinner(f"Chargement des données… (days={days})"):
                    df = fetch_data(
                        days=days,
                        coin_id=getattr(config, "DEFAULT_COIN_ID", "bitcoin"),
                        vs_currency=getattr(config, "DEFAULT_VS_CURRENCY", "usd"),
                    )
                    df = add_indicators(df)
                prev_days = days

            # 3) évaluation de la configuration
            pf, metrics, trades = _evaluate_run(df, params)
            row = {**params, **metrics}
            results.append(row)

            # 3bis) sauvegarde incrémentale (append 1 ligne)
            pd.DataFrame([row]).to_csv(
                csv_live,
                mode="a",
                header=not os.path.exists(csv_live),
                index=False
            )
            # Snapshot du meilleur en cours (utile si la session coupe)
            if best is None:
                best_snapshot = {"tested": i, "total": len(combos)}
            else:
                best_snapshot = {
                    "tested": i,
                    "total": len(combos),
                    "best_score": float(best[0]),
                    "best_params": best[1],
                }
            with open(os.path.join(out_dir, "best_so_far.json"), "w") as f:
                json.dump(best_snapshot, f, indent=2)

            # (optionnel) détails par run
            if getattr(config, "SAVE_PER_RUN_DETAILS", False):
                prefix = os.path.join(out_dir, f"run_{i:04d}")
                pf.to_csv(f"{prefix}_portfolio.csv", index=False)

            score = metrics["final_value"]  # critère = valeur finale max
            if (best is None) or (score > best[0]):
                best = (score, params, pf, trades)
                best_days = days

            progress.progress(int(100 * i / len(combos)))

        # Tableau de TOUTES les configurations
        res_df = pd.DataFrame(results).sort_values(by="final_value", ascending=False)
        st.dataframe(res_df, use_container_width=True)

        # Recharger le dataset correspondant au meilleur run si besoin
        if best_days != prev_days or df is None:
            with st.spinner(f"Rechargement des données pour le meilleur run… (days={best_days})"):
                df_best = fetch_data(
                    days=best_days,
                    coin_id=getattr(config, "DEFAULT_COIN_ID", "bitcoin"),
                    vs_currency=getattr(config, "DEFAULT_VS_CURRENCY", "usd"),
                )
                df_best = add_indicators(df_best)
        else:
            df_best = df

        # Sauvegardes finales (dans le même out_dir)
        _save_results(res_df, best, df_best, out_dir)

        # Affichage du meilleur run
        _, best_params, best_pf, (buys_d, buys_p, sells_d, sells_p) = best
        st.subheader("🏆 Meilleur jeu de paramètres")
        st.json(best_params)

        # Graphe principal (Buy&Hold vs portefeuille) avec le bon df
        series = [
            ("Buy & Hold", (df_best['price'] / df_best['price'].iloc[0]) * float(best_pf['portfolio_value'].iloc[0])),
            ("RandomForest", best_pf['portfolio_value']),
        ]
        st.plotly_chart(render_main_chart(df_best, series), use_container_width=True)

        # Graphe des trades
        st.plotly_chart(
            render_trades_chart(df_best, buys_d, buys_p, sells_d, sells_p, title="Trades (meilleur run)"),
            use_container_width=True
        )
        return  # fin du mode AUTO_TEST


    # ------- MODE INTERACTIF (inchangé) -------
    opts = render_sidebar()
    with st.spinner("Chargement des données…"):
        df = fetch_data(days=opts.days, coin_id=opts.coin_id, vs_currency=opts.vs_currency)
        df = add_indicators(df)

    with st.spinner("Simulation RandomForest…"):
        pf_rf, (buys_d, buys_p, sells_d, sells_p) = _call_simulate(
            df,
            {"n_lags": opts.n_lags, "horizon": opts.horizon, "train_step": opts.train_step},
            want_signals=True
        )

    series = [
        ("Buy & Hold", (df['price'] / df['price'].iloc[0]) * 10000),
        ("RandomForest", pf_rf['portfolio_value'])
    ]
    st.plotly_chart(render_main_chart(df, series), use_container_width=True)

    # Derniers montants (arrondis pour éviter les micro-restes)
    last = pf_rf.iloc[-1]
    price_now = float(df['price'].iloc[-1])

    cash_now = float(last.get('cash', 0.0))
    btc_now = float(last.get('btc', 0.0))
    state_now = str(last.get('state', 'cash'))

    if abs(cash_now) < 1e-8: cash_now = 0.0
    if abs(btc_now) < 1e-12: btc_now = 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("État", "CASH" if state_now == "cash" else "CRYPTO")
    col2.metric("Valeur portefeuille", f"${last['portfolio_value']:,.2f}")
    col3.metric("Cash", f"${cash_now:,.2f}")
    col4.metric("BTC", f"{btc_now:.6f} BTC (~${btc_now*price_now:,.2f})")

    if opts.show_trades:
        st.plotly_chart(
            render_trades_chart(df, buys_d, buys_p, sells_d, sells_p, title="Trades (RandomForest)"),
            use_container_width=True
        )


if __name__ == "__main__":
    main()
