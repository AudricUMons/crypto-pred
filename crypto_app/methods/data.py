# crypto_app/methods/data.py
from __future__ import annotations
import os
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests

# =========================
#  CONFIG OFFLINE KAGGLE
# =========================
# Dossier du dataset Kaggle "market-data" (peut être surchargé par une variable d'env)
KAGGLE_MARKET_DIR = os.getenv("KAGGLE_MARKET_DIR", "/kaggle/input/market-data")

# Mappage basique pour symboles
_SYMBOLS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "ripple": "XRP",
    "dogecoin": "DOGE",
    "cardano": "ADA",
    "litecoin": "LTC",
    "polkadot": "DOT",
    "chainlink": "LINK",
    "tron": "TRX",
}

def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def _days_ago_ms(days: int) -> int:
    return int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

# ===========================================================
#   Providers en ligne (réseau) — utilisés hors Kaggle
#   (inchangés, mais gardés ici pour fallback local/GA)
# ===========================================================

def _binance_hist_1h(days: int, coin_id: str, vs_currency: str) -> pd.DataFrame:
    base = "https://api.binance.com"
    sym = _SYMBOLS.get(coin_id.lower(), coin_id.upper())
    quote = "USDT" if vs_currency.lower() == "usd" else "EUR"
    symbol = f"{sym}{quote}"

    start = _days_ago_ms(days)
    end   = _now_ms()
    limit = 1000
    url = f"{base}/api/v3/klines"
    out_rows = []

    while True:
        params = {"symbol": symbol, "interval": "1h", "startTime": start, "endTime": end, "limit": limit}
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            raise RuntimeError("Binance rate limited (429)")
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out_rows.extend(data)
        last_close = data[-1][6]  # closeTime ms
        next_start = last_close + 1
        if next_start >= end or len(data) < limit:
            break
        start = next_start
        time.sleep(0.05)

    if not out_rows:
        raise RuntimeError(f"Aucune donnée Binance pour {symbol}")

    df = pd.DataFrame(out_rows, columns=[
        "openTime","open","high","low","close","volume","closeTime",
        "qav","trades","taker_base","taker_quote","ignore"
    ])
    df["date"] = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    df["price"] = df["close"].astype(float)
    return df[["date","price"]].set_index("date").sort_index()

def _coincap_hist_1h(days: int, coin_id: str, vs_currency: str) -> pd.DataFrame:
    base = "https://api.coincap.io/v2"
    start = _days_ago_ms(days)
    end   = _now_ms()
    url = f"{base}/assets/{coin_id.lower()}/history"
    params = {"interval": "h1", "start": start, "end": end}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 429:
        raise RuntimeError("CoinCap rate limited (429)")
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise RuntimeError(f"Aucune donnée CoinCap pour {coin_id}")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["price"] = df["priceUsd"].astype(float)
    # Conversion FX ignorée (dataset Kaggle déjà en USD)
    return df[["date","price"]].set_index("date").sort_index()

def _cryptocompare_hist_1h(days: int, coin_id: str, vs_currency: str) -> pd.DataFrame:
    fsym = _SYMBOLS.get(coin_id.lower(), coin_id.upper())
    tsym = "USD" if vs_currency.lower() == "usd" else "EUR"
    hours = days * 24
    url = "https://min-api.cryptocompare.com/data/v2/histohour"

    out = []
    to_ts = None
    remaining = hours
    while remaining > 0:
        batch = min(remaining, 2000)
        params = {"fsym": fsym, "tsym": tsym, "limit": batch - 1}
        if to_ts:
            params["toTs"] = to_ts
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            raise RuntimeError("CryptoCompare rate limited (429)")
        r.raise_for_status()
        payload = r.json()
        if payload.get("Response") != "Success":
            raise RuntimeError(f"CryptoCompare error: {payload.get('Message')}")
        data = payload["Data"]["Data"]
        if not data:
            break
        out = data[:-1] + out
        to_ts = data[0]["time"]
        remaining -= len(data)
        time.sleep(0.05)

    if not out:
        raise RuntimeError(f"Aucune donnée CryptoCompare pour {fsym}/{tsym}")

    df = pd.DataFrame(out)
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["price"] = df["close"].astype(float)
    return df[["date","price"]].set_index("date").sort_index()

# ===========================================================
#                 API Publique: fetch_data
#   -> Kaggle offline si CSV présent, sinon fetch online
# ===========================================================

def fetch_data(days: int = 90, coin_id: str = "bitcoin", vs_currency: str = "usd",
               provider: str = "auto") -> pd.DataFrame:
    """
    Retourne un DataFrame indexé par datetime (UTC) avec colonne 'price'.
    Sur Kaggle (offline), lit /kaggle/input/market-data/btc_usd_{days}d.csv
    si disponible. Sinon, fallback vers les providers online (local/GA).
    """
    # --- 1) MODE KAGGLE OFFLINE (CSV déjà fourni par GitHub Action) ---
    if os.path.isdir(KAGGLE_MARKET_DIR):
        csv_name = f"btc_usd_{int(days)}d.csv"
        csv_path = os.path.join(KAGGLE_MARKET_DIR, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # On accepte soit 'ts' soit 'date' dans le CSV
            time_col = "ts" if "ts" in df.columns else ("date" if "date" in df.columns else None)
            if time_col is None:
                raise ValueError(f"Le CSV {csv_name} doit contenir 'ts' ou 'date'.")
            df["date"] = pd.to_datetime(df[time_col], utc=True)
            if "price" not in df.columns:
                raise ValueError(f"Le CSV {csv_name} doit contenir une colonne 'price'.")
            return df[["date","price"]].set_index("date").sort_index()

    # --- 2) MODE ONLINE (local / GitHub Actions) ---
    providers = ["binance", "coincap", "cryptocompare"] if provider == "auto" else [provider]
    last_err = None
    for p in providers:
        try:
            if p == "binance":
                return _binance_hist_1h(days, coin_id, vs_currency)
            if p == "coincap":
                return _coincap_hist_1h(days, coin_id, vs_currency)
            if p == "cryptocompare":
                return _cryptocompare_hist_1h(days, coin_id, vs_currency)
            raise ValueError(f"Provider inconnu: {p}")
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Aucun provider n'a répondu (dernier: {last_err})")
