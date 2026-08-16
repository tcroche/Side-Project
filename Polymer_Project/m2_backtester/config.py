# config.py

# --- Marché / session ---
MINUTES_PER_DAY = 390      # session actions US (à adapter selon l'indice réel)
TRADING_DAYS_PER_YEAR = 252

# --- Frais de transaction (consignes : basis_point = 0.0001 = 1 bp) ---
BASIS_POINT = 0.0001

# --- Découpage In-Sample / Out-of-Sample (consignes : 5 mois IS / 1 mois OOS) ---
IS_RATIO = 5 / 6           # fraction des jours utilisée pour le tuning

# --- Données ---
DATA_DIR = "data"          # sous-dossier où sont stockés les .pkl

# --- Reproductibilité ---
RANDOM_SEED = 42

# --- Univers de travail : indices boursiers mondiaux (cf. consignes) ---
# Cles NORMALISEES (sans ^ ni =). Le loader fait la correspondance.
UNIVERSE = ["GSPC", "DJI", "RUT", "FTSE", "N225"]   # S&P500, Dow, Russell2000, FTSE100, Nikkei
TICKERS = UNIVERSE

# Fichiers a ignore
NON_INSTRUMENTS = {"PNL", "DAILY"}

# --- Decoupage IS / OOS par date (donnees : 2025-01-06 -> 2025-07-31) ---
OOS_START = "2025-06-01"   # IS = avant cette date ; OOS = a partir de cette date

# --- Parametres de strategie (valeurs de depart, Elles seront calibrer duran l'IS) ---
STRAT_WINDOW = 30        # minutes pour la moyenne/ecart-type glissants
STRAT_K_ENTRY = 1.5      # seuil d'entree (en z-score)
STRAT_K_EXIT = 0.5       # seuil de sortie (en z-score)
STRAT_STOP_LOSS = 0.005  # stop-loss : 0.5% de perte relative (None pour desactiver)
