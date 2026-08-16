# backtester.py

import numpy as np
import pandas as pd

import config
import data_loader
import strategy


# =========================================================================
#  1) BACKTEST D'UNE JOURNEE (un actif x un jour) -> analogue de backtest()
# =========================================================================

def backtest_day(close, params=None):
    """Rejoue UNE session intraday d'un actif.
    'close' : pd.Series des prix de cloture 1-min (index Datetime), une journee.
    'params': dict optionnel {window, k_entry, k_exit, stop_loss} pour surcharger
              les valeurs de config (utile pour la calibration IS).
    Retourne un dict de resultats journaliers, ou None si la journee est trop courte."""
    close = close.dropna()
    n = len(close)
    if n < 3:
        return None

    pos = strategy.compute_positions(close, **(params or {})).values
    P = close.values

    # --- PnL brut : pnl_t = pos[t]*(P[t+1]-P[t]) ---------------------------
    dP = np.diff(P)                 # P[t+1]-P[t], longueur n-1
    pos_held = pos[:-1]             # position effectivement detenue sur [t, t+1]
    gross_pnl_bar = pos_held * dP
    gross_pnl = float(gross_pnl_bar.sum())

    # --- Frais : une jambe a chaque changement de position -----------------
    prev_pos = np.concatenate([[0.0], pos[:-1]])   # position a la barre precedente (0 avant l'ouverture)
    dpos = pos - prev_pos                           # quantite tradee a la barre t
    fees_bar = config.BASIS_POINT * np.abs(dpos) * P
    total_fees = float(fees_bar.sum())

    # --- PnL net + conversion en rendement journalier ---------------------
    net_pnl = gross_pnl - total_fees
    base = float(P[0])              # notionnel deploye = 1 unite de l'indice (prix d'ouverture)
    gross_ret = gross_pnl / base
    net_ret = net_pnl / base

    # --- Comptage des trades (entrees = passages flat -> en position) ------
    num_entries = int(((prev_pos == 0) & (pos != 0)).sum())

    return {
        "grossPnL": gross_pnl,
        "feesTrade": total_fees,
        "netPnL": net_pnl,
        "grossRet": gross_ret,     # rendement brut du jour (pour le portefeuille)
        "netRet": net_ret,         # rendement net du jour (pour le portefeuille / Sharpe)
        "numTrade": num_entries,   # nb d'aller-retours inities ce jour
        "nBars": int(n),
    }


# =========================================================================
#  2) BACKTEST D'UN ACTIF SUR TOUTES SES JOURNEES
# =========================================================================

def run_backtest_ticker(ticker, df, params=None, start=None, end=None):
    """Boucle jour par jour sur un actif. 'start'/'end' sont des datetime.date
    optionnels pour restreindre la periode (decoupage IS/OOS)."""
    rows = []
    df = df.sort_index()
    for day_ts, day_df in df.groupby(df.index.normalize()):
        d = day_ts.date()
        if start is not None and d < start:
            continue
        if end is not None and d > end:
            continue
        res = backtest_day(day_df["Close"], params)
        if res is None:
            continue
        res["Date"] = pd.Timestamp(d)
        res["Ticker"] = ticker
        rows.append(res)
    return pd.DataFrame(rows)


# =========================================================================
#  3) BACKTEST DE TOUT L'UNIVERS -> analogue de core_run_backtest
# =========================================================================

def run_backtest(series=None, params=None, universe=config.UNIVERSE,
                 start=None, end=None):
    """Renvoie un DataFrame 'long' : une ligne = un actif x un jour.
    Colonnes : Date, Ticker, grossPnL, numTrade, feesTrade, netPnL, grossRet, netRet, nBars."""
    if series is None:
        series = data_loader.load_ticker_series()
    frames = []
    for tk in (universe if universe is not None else sorted(series)):
        if tk not in series:
            continue
        frames.append(run_backtest_ticker(tk, series[tk], params, start, end))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    cols = ["Date", "Ticker", "grossPnL", "numTrade", "feesTrade",
            "netPnL", "grossRet", "netRet", "nBars"]
    return out[cols].sort_values(["Ticker", "Date"]).reset_index(drop=True)


if __name__ == "__main__":
    series = data_loader.load_ticker_series()
    res = run_backtest(series)

    print(f"=== Backtest : {len(res)} lignes (actif x jour) ===\n")

    # Resume par actif : PnL brut/net cumule, frais, trades/jour moyen
    print(f"{'Ticker':8} {'jours':>6} {'grossPnL':>12} {'fees':>10} "
          f"{'netPnL':>12} {'tr/j':>6}")
    for tk, g in res.groupby("Ticker"):
        print(f"{tk:8} {len(g):>6} {g['grossPnL'].sum():>12.2f} "
              f"{g['feesTrade'].sum():>10.2f} {g['netPnL'].sum():>12.2f} "
              f"{g['numTrade'].mean():>6.1f}")

    # Coherence anti-look-ahead : la 1ere journee d'un actif
    tk0 = res["Ticker"].iloc[0]
    g0 = res[res["Ticker"] == tk0].iloc[0]
    print(f"\nExemple {tk0} {g0['Date'].date()} : "
          f"gross={g0['grossPnL']:.3f}  fees={g0['feesTrade']:.3f}  "
          f"net={g0['netPnL']:.3f}  netRet={g0['netRet']*100:.3f}%  "
          f"trades={int(g0['numTrade'])}")

    # Verification : frais > 0 des qu'il y a des trades ; net = gross - fees
    assert np.allclose(res["netPnL"], res["grossPnL"] - res["feesTrade"]), "net != gross - fees"
    assert (res.loc[res["numTrade"] > 0, "feesTrade"] > 0).all(), "trades sans frais ?"
    print("\nChecks OK : net = gross - fees, et tout trade paie des frais.")
