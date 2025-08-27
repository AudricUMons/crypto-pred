# scripts/fetch_and_export.py
import os
import sys
from pathlib import Path
import pandas as pd

# >>> Rendez le package visible même si PYTHONPATH n'est pas réglé
ROOT = Path(__file__).resolve().parents[1]  # racine du repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crypto_app.methods.data import fetch_data  # import fonctionne maintenant

COIN_ID = os.environ.get("COIN_ID", "bitcoin")
VS_CCY  = os.environ.get("VS_CCY", "usd")
OUTDIR  = "data_export"

def dump(days: int):
    df = fetch_data(days=days, coin_id=COIN_ID, vs_currency=VS_CCY, provider="auto")
    if df is None or len(df) == 0:
        raise SystemExit(f"ERROR: fetch_data returned empty for {days}d")

    # Normalise en ts/price
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True)
        price = df["price"].astype(float) if "price" in df.columns else None
    else:
        idx = getattr(df, "index", None)
        if idx is None or len(df) == 0:
            raise SystemExit(f"ERROR: empty index for {days}d")
        ts = pd.to_datetime(idx, utc=True)
        cand = [c for c in ("price", "close", "Close", "Adj Close", "adj_close") if c in df.columns]
        if not cand:
            raise SystemExit(f"ERROR: no price-like column for {days}d")
        price = df[cand[0]].astype(float)

    out = pd.DataFrame({"ts": ts, "price": price}).dropna()
    if out.empty:
        raise SystemExit(f"ERROR: empty after normalization for {days}d")

    out = out.sort_values("ts").drop_duplicates("ts")
    os.makedirs(OUTDIR, exist_ok=True)
    out.to_csv(os.path.join(OUTDIR, f"btc_usd_{days}d.csv"), index=False)
    print(f"Wrote {OUTDIR}/btc_usd_{days}d.csv ({len(out)} rows)")

def main():
    for d in (60, 90, 120):
        dump(d)
    print("DONE")

if __name__ == "__main__":
    sys.exit(main())
