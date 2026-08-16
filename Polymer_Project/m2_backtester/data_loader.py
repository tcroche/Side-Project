# data_loader.py

import os
import glob
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config


# =========================================================================
#  1) CHARGEMENT DES DONNEES
# =========================================================================

def flatten_columns(df):
    """Aplatit les colonnes MultiIndex ('Close','^GSPC') -> 'Close'.
    N'agit QUE si c'est un MultiIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def normalize_ticker(raw):
    """Fusionne les alias de nommage : enleve '^' et '=', met en majuscules.
    '^GSPC'->'GSPC', 'BZ=F'->'BZF', 'EURUSD=X'->'EURUSDX'."""
    return raw.replace('^', '').replace('=', '').upper()


def ticker_from_filename(path, token=1):
    return os.path.basename(path).split('_')[token]


def discover_files(root=config.DATA_DIR):
    for pattern in (os.path.join(root, "Yahoo_*", "*.pkl"),
                    os.path.join(root, "*.pkl"),
                    os.path.join(root, "**", "*.pkl")):
        files = glob.glob(pattern, recursive=True)
        if files:
            return sorted(files)
    return []


def load_ticker_series(root=config.DATA_DIR, universe=config.UNIVERSE):
    """Charge et fusionne les fichiers par instrument NORMALISE.
    - aplatit CHAQUE fichier avant concat (evite le desalignement de colonnes)
    - regroupe les alias ('^GSPC' et 'GSPC' -> 'GSPC')
    - ignore les fichiers parasites (pnl, daily)
    - filtre sur 'universe' (passer universe=None pour tout charger)
    Renvoie {ticker: DataFrame continu indexe par Datetime}."""
    files = discover_files(root)
    if not files:
        raise FileNotFoundError(
            f"Aucun .pkl trouve sous '{root}'. Place les donnees dedans."
        )

    by_ticker = {}
    for f in files:
        tk = normalize_ticker(ticker_from_filename(f))
        if tk in config.NON_INSTRUMENTS:
            continue
        if universe is not None and tk not in universe:
            continue
        df = flatten_columns(pd.read_pickle(f))     # APLATIR PAR FICHIER (important)
        by_ticker.setdefault(tk, []).append(df)

    out = {}
    for tk, parts in by_ticker.items():
        s = pd.concat(parts)
        s = s[~s.index.duplicated(keep="first")].sort_index()
        if "Close" in s.columns:
            s["Close"] = s["Close"].ffill()
            s = s.dropna(subset=["Close"])
        out[tk] = s
    return out


# =========================================================================
#  2) GENERATEUR SYNTHETIQUE 
# =========================================================================

def generate_synthetic_day(ticker, day, n_minutes=config.MINUTES_PER_DAY,
                           start_price=100.0, annual_vol=0.20, seed=None):
    rng = np.random.default_rng(seed)
    sigma_min = annual_vol / np.sqrt(config.MINUTES_PER_DAY * config.TRADING_DAYS_PER_YEAR)
    log_returns = rng.normal(0.0, sigma_min, size=n_minutes)
    close = start_price * np.exp(np.cumsum(log_returns))
    open_ = np.empty(n_minutes); open_[0] = start_price; open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, sigma_min * start_price, size=n_minutes))
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = rng.integers(1_000, 10_000, size=n_minutes).astype(float)
    start_ts = pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=30)
    index = pd.date_range(start=start_ts, periods=n_minutes, freq="1min", name="Datetime")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=index)


def trading_days(start, n_days):
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def generate_synthetic_dataset(tickers=config.TICKERS, n_days=126,
                               out_dir=config.DATA_DIR, seed=config.RANDOM_SEED):
    os.makedirs(out_dir, exist_ok=True)
    days = trading_days(date(2025, 1, 1), n_days)
    count = 0
    for t_idx, ticker in enumerate(tickers):
        prev_close = 100.0 * (1 + t_idx)
        for d_idx, day in enumerate(days):
            df = generate_synthetic_day(ticker, day, start_price=prev_close,
                                        seed=seed + t_idx * 10_000 + d_idx)
            prev_close = float(df["Close"].iloc[-1])
            df.to_pickle(os.path.join(out_dir, f"df_{ticker}_{day.strftime('%Y%m%d')}.pkl"))
            count += 1
    print(f"{count} fichiers synthetiques generes dans '{out_dir}'.")


if __name__ == "__main__":
    series = load_ticker_series()      # filtre automatiquement sur config.UNIVERSE
    print(f"=== {len(series)} indices charges : {sorted(series)} ===\n")
    print(f"{'Ticker':8} {'Barres':>8} {'Jours':>6}  Periode")
    for tk, df in sorted(series.items()):
        days = df.index.normalize().nunique()
        print(f"{tk:8} {len(df):>8} {days:>6}  "
              f"{df.index.min().date()} -> {df.index.max().date()}")
