# crypto_app/methods/data.py
from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
import math
import pandas as pd
import requests

# --- Helpers ---------------------------------------------------------------

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

# --- Provider 1: Binance (klines 1h, sans clé) ----------------------------

def _binance_hist_1h(days: int, coin_id: str, vs_currency: str) -> pd.DataFrame:
    base = "https://api.binance.com"
    sym = _SYMBOLS.get(coin_id.lower(), coin_id.upper())   # ex: bitcoin -> BTC
    quote = "USDT" if vs_currency.lower() == "usd" else "EUR"
    symbol = f"{sym}{quote}"

    start = _days_ago_ms(days)
    end   = _now_ms()
    limit = 1000  # max points par requête
    url = f"{base}/api/v3/klines"
    out_rows = []

    while True:
        params = {"symbol": symbol, "interval": "1h", "startTime": start, "endTime": end, "limit": limit}
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            # rate limit → on stoppe et on laissera le fallback prendre la main
            raise RuntimeError("Binance rate limited (429)")
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out_rows.extend(data)
        last_close = data[-1][6]  # closeTime en ms
        next_start = last_close + 1
        if next_start >= end or len(data) < limit:
            break
        start = next_start
        time.sleep(0.05)  # pause légère

    if not out_rows:
        raise RuntimeError(f"Aucune donnée Binance pour {symbol}")

    # kline: [openTime, open, high, low, close, volume, closeTime, ...]
    df = pd.DataFrame(out_rows, columns=[
        "openTime","open","high","low","close","volume","closeTime",
        "qav","trades","taker_base","taker_quote","ignore"
    ])
    df["date"] = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    df["price"] = df["close"].astype(float)
    df = df[["date","price"]].set_index("date").sort_index()
    return df

# --- Provider 2: CoinCap (h1, sans clé) -----------------------------------

def _coincap_hist_1h(days: int, coin_id: str, vs_currency: str) -> pd.DataFrame:
    # CoinCap renvoie priceUsd uniquement -> on convertit si vs_currency != usd (minimaliste)
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
    df = df[["date","price"]].set_index("date").sort_index()
    if vs_currency.lower() != "usd":
        # Conversion naïve (pas de FX ici). Pour l’EUR, c’est approximatif.
        pass
    return df

# --- Provider 3: CryptoCompare (histohour, sans clé) ----------------------

def _cryptocompare_hist_1h(days: int, coin_id: str, vs_currency: str) -> pd.DataFrame:
    fsym = _SYMBOLS.get(coin_id.lower(), coin_id.upper())
    tsym = "USD" if vs_currency.lower() == "usd" else "EUR"
    hours = days * 24
    url = "https://min-api.cryptocompare.com/data/v2/histohour"

    out = []
    # CryptoCompare limite 'limit' ~2000; on pagine en remontant avec toTs
    to_ts = None
    remaining = hours
    while remaining > 0:
        batch = min(remaining, 2000)  # points
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
        out = data[:-1] + out  # évite doublons en chevauchant
        to_ts = data[0]["time"]  # prochaine borne
        remaining -= len(data)
        time.sleep(0.05)

    if not out:
        raise RuntimeError(f"Aucune donnée CryptoCompare pour {fsym}/{tsym}")

    df = pd.DataFrame(out)
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["price"] = df["close"].astype(float)
    df = df[["date","price"]].set_index("date").sort_index()
    return df

# --- API publique de haut niveau ------------------------------------------

def fetch_data(days: int = 90, coin_id: str = "bitcoin", vs_currency: str = "usd",
               provider: str = "auto") -> pd.DataFrame:
    """
    Essaie plusieurs sources gratuites d'OHLC 1h (sans clé):
    - binance (BTCUSDT / BTCEUR…)
    - coincap (assets/{id}/history)
    - cryptocompare (histohour)
    Retourne un DataFrame indexé par datetime UTC, col 'price'.
    """
    providers = []
    if provider == "auto":
        providers = ["binance", "coincap", "cryptocompare"]
    else:
        providers = [provider]

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
            # on tente le provider suivant
            continue
    raise RuntimeError(f"Aucun provider gratuit n'a répondu. Dernière erreur: {last_err}")
