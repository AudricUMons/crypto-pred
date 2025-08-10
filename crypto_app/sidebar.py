
from dataclasses import dataclass
import streamlit as st
from .config import DEFAULT_DAYS, DEFAULT_COIN_ID, RF_N_LAGS, RF_HORIZON, RF_TRAIN_STEP

@dataclass
class Options:
    coin_id: str
    vs_currency: str
    days: int
    n_lags: int
    horizon: int
    train_step: int
    show_trades: bool

def render_sidebar() -> Options:
    st.sidebar.header("Options")
    coin = st.sidebar.text_input("CoinGecko coin id", value=DEFAULT_COIN_ID)
    vs   = st.sidebar.selectbox("Devise", ["usd","eur"], index=0)
    days = st.sidebar.slider("Période (jours)", 30, 365, value=DEFAULT_DAYS)
    n_lags = st.sidebar.slider("RF: n_lags", 3, 48, value=RF_N_LAGS)
    horizon = st.sidebar.slider("RF: horizon (heures)", 1, 24, value=RF_HORIZON)
    train_step = st.sidebar.slider("RF: retrain chaque n pas", 1, 12, value=RF_TRAIN_STEP)
    show_trades = st.sidebar.checkbox("Afficher les trades RF", value=True)
    return Options(coin_id=coin, vs_currency=vs, days=days,
                   n_lags=n_lags, horizon=horizon, train_step=train_step,
                   show_trades=show_trades)
