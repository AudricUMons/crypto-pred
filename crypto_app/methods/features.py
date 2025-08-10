
import pandas as pd
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['return'] = out['price'].pct_change().fillna(0.0)
    out['MA7']  = out['price'].rolling(7).mean()
    out['MA30'] = out['price'].rolling(30).mean()
    out['volatility'] = out['return'].rolling(24).std().fillna(0.0)
    return out
