#calibrate_is.py

import datetime as dt
import itertools
import numpy as np
import pandas as pd

import config
import data_loader
from m2_backtester import backtester, portfolio
import strategy_momentum

OOS_START = dt.date.fromisoformat(config.OOS_START)
IS_END = OOS_START - dt.timedelta(days=1)

WINDOWS = [90,120,150]
K_ENTRYS = [2.5, 3.0, 3.5]
STOPS = [0.005, 0.01]
K_EXIT   = 0.3       # fixe (valeur figee)
GAMMA    = 0.3       # fixe pour la grille principale (balayage separe ensuite)

FROZEN = dict(window=120, k_entry=3.0, stop_loss=0.005)   # cellule pre-engagee


def ew_sharpe(ret, exclude=()):
    """Sharpe net + rendement net du portefeuille equal-weight (renormalise/jour)."""
    cols = [c for c in ret.columns if c not in exclude]
    if not cols:
        return np.nan, np.nan
    p = portfolio.portfolio_returns(ret[cols], portfolio.equal_weights(cols))
    return portfolio.ann_sharpe(p), 100.0 * p.sum()


def eval_cell(series, window, k_entry, stop_loss, gamma=GAMMA):
    params = dict(window=window, k_entry=k_entry, k_exit=K_EXIT,
                  stop_loss=stop_loss, gamma=gamma)
    backtester.strategy = strategy_momentum
    res = backtester.run_backtest(series, params=params, end=IS_END)
    ret = portfolio.returns_matrix(res, "netRet")
    s_all, r_all = ew_sharpe(ret)
    s_no, r_no = ew_sharpe(ret, exclude=("RUT",))
    trj = float(res["numTrade"].mean()) if not res.empty else np.nan
    return dict(window=window, k_entry=k_entry, stop=stop_loss,
                sharpe_RUT=s_all, ret_RUT=r_all,
                sharpe_noRUT=s_no, ret_noRUT=r_no, trj=trj)


if __name__ == "__main__":
    series = data_loader.load_ticker_series()
    print("Calcul en cours\n")

    rows = [eval_cell(series, w, ke, sl)
            for w, ke, sl in itertools.product(WINDOWS, K_ENTRYS, STOPS)]
    grid = pd.DataFrame(rows)

    is_frozen = ((grid["window"] == FROZEN["window"]) &
                 (grid["k_entry"] == FROZEN["k_entry"]) &
                 (grid["stop"] == FROZEN["stop_loss"]))
    grid["fige"] = np.where(is_frozen, "*", "")

    # --- 1) Tableau complet ---
    print("=== GRILLE DE ROBUSTESSE (IS) — momentum faible rotation ===")
    print(f"{'win':>4}{'k_ent':>7}{'stop':>7}{'Sh+RUT':>9}{'ret+%':>8}"
          f"{'Sh-RUT':>9}{'ret-%':>8}{'tr/j':>6}{'':>3}")
    for _, r in grid.iterrows():
        print(f"{int(r['window']):>4}{r['k_entry']:>7.1f}{r['stop']:>7.3f}"
              f"{r['sharpe_RUT']:>9.2f}{r['ret_RUT']:>8.2f}"
              f"{r['sharpe_noRUT']:>9.2f}{r['ret_noRUT']:>8.2f}"
              f"{r['trj']:>6.1f}{r['fige']:>3}")

    # --- 2) La cellule figee est-elle typique ou un pic ? ---
    def summarize(col, label):
        s = grid[col]
        fro = float(grid.loc[is_frozen, col].iloc[0])
        pct_pos = 100.0 * (s > 0).mean()
        rank = 100.0 * (s < fro).mean()       # percentile de la cellule figee
        print(f"\n--- {label} ---")
        print(f"  mediane={s.median():.2f}  min={s.min():.2f}  max={s.max():.2f}")
        print(f"  cellules Sharpe>0 : {pct_pos:.0f}%  ({int((s > 0).sum())}/{len(s)})")
        print(f"  cellule figee = {fro:.2f}  -> percentile {rank:.0f}% de la grille")
        if fro >= s.max() - 1e-9:
            print("  ATTENTION : la cellule figee est le PIC -> signal de fragilite.")
        elif rank <= 70 and pct_pos >= 60:
            print("  -> figee TYPIQUE du voisinage et majorite positive : ROBUSTE.")
        else:
            print("  -> a interpreter avec prudence (voir distribution ci-dessus).")

    summarize("sharpe_RUT", "Sharpe AVEC RUT")
    summarize("sharpe_noRUT", "Sharpe SANS RUT")

    # --- 3) Invariance du Sharpe a gamma (sizing fige a l'entree = levier) ---
    print("\n=== INVARIANCE A GAMMA (cellule figee) ===")
    print(f"{'gamma':>6}{'Sh+RUT':>9}{'Sh-RUT':>9}{'ret+%':>8}{'ret-%':>8}")
    for g in [0.2, 0.3, 0.5]:
        c = eval_cell(series, FROZEN["window"], FROZEN["k_entry"],
                      FROZEN["stop_loss"], gamma=g)
        print(f"{g:>6.2f}{c['sharpe_RUT']:>9.2f}{c['sharpe_noRUT']:>9.2f}"
              f"{c['ret_RUT']:>8.2f}{c['ret_noRUT']:>8.2f}")
    print("(Sharpe quasi-identique sur gamma -> confirme : le sizing tanh fige a "
          "l'entree n'est qu'un levier, pas un createur d'edge.)")

    print(f"\nConfigurations testees cette phase : {len(grid)} (grille) + 3 (gamma) = "
          f"{len(grid) + 3}.  A cumuler avec les explorations precedentes pour le "
          f"Deflated Sharpe (rapport).")
