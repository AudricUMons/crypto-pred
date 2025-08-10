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
PARAM_GRID = {
    # Données
    "days":        [60, 90, 120],            # <-- différent de DEFAULT_DAYS

    # Modèle
    "n_lags":       [24, 27, 30, 33, 36],    # <-- différent de RF_N_LAGS
    "horizon":      [18, 21, 24, 27, 30],    # <-- différent de RF_HORIZON
    "n_estimators": [64, 128, 256],          # <-- différent de RF_N_ESTIMATORS
    "train_step":   [1, 2],                  # <-- différent de RF_TRAIN_STEP
    "threshold":    [0.0010, 0.0015, 0.0020, 0.0030],  # <-- différent de RF_THRESHOLD
    "fee_rate":     [0.001],                 # tu peux élargir si besoin
}
