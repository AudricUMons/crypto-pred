# =========================
#  Auto test au lancement
# =========================
AUTO_TEST = True   # Mettre False pour repasser en mode interactif

# =========================
#  Données par défaut (UI)
# =========================
DEFAULT_DAYS = 90                 # historique par défaut
DEFAULT_COIN_ID = "bitcoin"       # 'bitcoin', 'ethereum', ...
DEFAULT_VS_CURRENCY = "usd"       # 'usd' ou 'eur'

# =========================
#  Hyperparamètres par défaut (UI)
#  -> utilisés en mode interactif et comme "centre" du grid
# =========================
RF_N_LAGS = 30
RF_HORIZON = 24
RF_N_ESTIMATORS = 128
RF_TRAIN_STEP = 1
RF_THRESHOLD = 0.002              # 0.2% ~ aggressivité raisonnable
RF_FEE_RATE = 0.001               # 0.1% de frais

# =========================
#  Sauvegardes
# =========================
SAVE_RESULTS = True               # écrit un CSV de toutes les configs
SAVE_PER_RUN_DETAILS = False      # True => CSV portefeuille + trades de CHAQUE run (lourd)
RESULTS_DIR = "results"           # sur Colab: "/content/drive/MyDrive/results"

# =========================
#  GRID d'exploration AUTO_TEST
#  Note: 'days' est pris en charge si tu appliques le patch app.py ci-dessous.
# =========================
PARAM_GRID = [
    # 1) Jours = 60, n_lags = 24 — SEULEMENT les horizons non testés / combos restants
    #    Bilan: pour horizon=27 il reste 14 combos (voir sous-blocs) ; pour horizon=30 il reste tout (24 combos).
    #    Total block(1): 38 combos.
    # ---- horizon = 27
    {
        "days": [60],
        "n_lags": [24],
        "horizon": [27],
        "n_estimators": [128],
        "train_step": [1],
        "threshold": [0.0020, 0.0030],      # (0.0010 et 0.0015 déjà testés en step=1)
        "fee_rate": [0.001],
    },
    {
        "days": [60],
        "n_lags": [24],
        "horizon": [27],
        "n_estimators": [128],
        "train_step": [2],
        "threshold": [0.0010, 0.0015, 0.0020, 0.0030],  # rien testé en step=2 pour 128
        "fee_rate": [0.001],
    },
    {
        "days": [60],
        "n_lags": [24],
        "horizon": [27],
        "n_estimators": [256],
        "train_step": [1, 2, 12, 24],
        "threshold": [0.0010, 0.0015, 0.0020, 0.0030],  # rien testé pour 256
        "fee_rate": [0.001],
    },
    # ---- horizon = 30 (rien testé → tout reste)
    {
        "days": [60],
        "n_lags": [24],
        "horizon": [30],
        "n_estimators": [64, 128, 256],
        "train_step": [1, 2, 12 , 24],
        "threshold": [0.0010, 0.0015, 0.0020, 0.0030],
        "fee_rate": [0.001],
    },

    # 2) Jours = 60, n_lags != 24 — (rien testé ici) → tout reste
    {
        "days": [60],
        "n_lags": [27, 30, 33, 36],
        "horizon": [18, 21, 24, 27, 30],
        "n_estimators": [64, 128, 256],
        "train_step": [1, 2, 12, 24],
        "threshold": [0.0010, 0.0015, 0.0020, 0.0030],
        "fee_rate": [0.001],
    },

    # 3) Jours = 90 ou 120 — (rien testé ici) → tout reste
    {
        "days": [90, 120],
        "n_lags": [24, 27, 30, 33, 36],
        "horizon": [18, 21, 24, 27, 30],
        "n_estimators": [64, 128, 256],
        "train_step": [1, 2, 12, 24],
        "threshold": [0.0010, 0.0015, 0.0020, 0.0030],
        "fee_rate": [0.001],
    },
]

