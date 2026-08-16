# metrics.py

import datetime as dt
import numpy as np
import pandas as pd

import config
import data_loader
from m2_backtester import backtester, portfolio
import strategy_momentum

OOS_START = dt.date.fromisoformat(config.OOS_START)   # 1er jour OOS (inclus)


# =====================================================================
#  1) ERREUR-TYPE DU SHARPE ANNUALISE
# =====================================================================

def sharpe_se(daily, ann=config.TRADING_DAYS_PER_YEAR):
    """SE approx. du Sharpe ANNUALISE :  sqrt(ann/n) * sqrt(1 + 0.5 * SR^2 / ann).
    Demonstration (Lo 2002) : pour des rendements iid, sqrt(T)*(SR_hat - SR) tend
    vers N(0, 1 + SR^2/2) ou SR est le Sharpe PAR PERIODE et T le nb de periodes.
    En annualisant (SR_ann = SR_periode * sqrt(ann)), on obtient la formule ci-dessus.
    Avec n ~ 40 jours, ce SE est enorme : c'est le chiffre qui prouve que l'IC du
    Sharpe OOS englobe largement zero."""
    a = np.asarray(daily, dtype=float)
    a = a[~np.isnan(a)]
    n = a.size
    sr = portfolio.ann_sharpe(a, ann)
    if n < 2 or np.isnan(sr):
        return np.nan
    return float(np.sqrt(ann / n) * np.sqrt(1.0 + 0.5 * sr ** 2 / ann))


# =====================================================================
#  2) METRIQUES PAR ACTIF
# =====================================================================

def per_asset_metrics(res):
    """Une ligne par actif : Gross%, Net%, Sharpe(net), MaxDD(net)%, trades/j.
    % = somme additive des rendements journaliers (coherent avec la courbe de
    capital additive utilisee partout dans le projet, cf. portfolio.max_drawdown)."""
    net = portfolio.returns_matrix(res, "netRet")
    gross = portfolio.returns_matrix(res, "grossRet")
    trj = res.groupby("Ticker")["numTrade"].mean()

    rows = []
    for tk in net.columns:
        n = net[tk].dropna()
        g = gross[tk].dropna()
        rows.append(dict(
            actif=tk,
            jours=len(n),
            gross_pct=100.0 * g.sum(),
            net_pct=100.0 * n.sum(),
            sharpe=portfolio.ann_sharpe(n),
            sharpe_se=sharpe_se(n),
            maxdd_pct=100.0 * portfolio.max_drawdown(n),
            trj=float(trj.get(tk, np.nan)),
        ))
    return pd.DataFrame(rows)


# =====================================================================
#  3) METRIQUES PORTEFEUILLE equal-weight
# =====================================================================

def portfolio_metrics(res, label, exclude=()):
    """Portefeuille equal-weight (renormalisation par jour des poids sur les
    actifs disponibles -> les feries JP du N225 ne biaisent rien)."""
    net = portfolio.returns_matrix(res, "netRet")
    gross = portfolio.returns_matrix(res, "grossRet")
    cols = [c for c in net.columns if c not in exclude]
    if not cols:
        return None
    w = portfolio.equal_weights(cols)

    p_net = portfolio.portfolio_returns(net[cols], w)
    p_gross = portfolio.portfolio_returns(gross[cols], w)

    # Trades/jour du PORTEFEUILLE = activite agregee du livre (somme des actifs
    # detenus ce jour-la), moyennee sur les jours de bourse. ATTENTION : les
    # lignes "par actif" ci-dessus sont des moyennes PAR actif -> non comparables
    # directement a cette ligne agregee (a preciser en note du tableau).
    sub = res[~res["Ticker"].isin(exclude)]
    daily_trades = sub.groupby("Date")["numTrade"].sum()

    return dict(
        actif=label,
        jours=len(p_net),
        gross_pct=100.0 * p_gross.sum(),
        net_pct=100.0 * p_net.sum(),
        sharpe=portfolio.ann_sharpe(p_net),
        sharpe_se=sharpe_se(p_net),
        maxdd_pct=100.0 * portfolio.max_drawdown(p_net),
        trj=float(daily_trades.mean()),
    )


# =====================================================================
#  4) AFFICHAGE
# =====================================================================

def print_matrix(df_assets, port_rows):
    head = (f"{'Actif':16}{'jours':>6}{'Gross%':>9}{'Net%':>9}"
            f"{'Sharpe':>8}{'MaxDD%':>9}{'tr/j':>7}")
    print(head)
    print("-" * len(head))
    for _, r in df_assets.iterrows():
        print(f"{r['actif']:16}{int(r['jours']):>6}{r['gross_pct']:>9.2f}"
              f"{r['net_pct']:>9.2f}{r['sharpe']:>8.2f}{r['maxdd_pct']:>9.2f}"
              f"{r['trj']:>7.1f}")
    print("-" * len(head))
    for r in port_rows:
        if r is None:
            continue
        print(f"{r['actif']:16}{int(r['jours']):>6}{r['gross_pct']:>9.2f}"
              f"{r['net_pct']:>9.2f}{r['sharpe']:>8.2f}{r['maxdd_pct']:>9.2f}"
              f"{r['trj']:>7.1f}")


def print_sharpe_ci(port_rows, z=1.96):
    """L'element-cle de la conclusion : l'IC 95% du Sharpe sur ~40 jours."""
    print("\n--- Sharpe OOS annualise + IC 95% (Lo 2002, hyp. iid) ---")
    for r in port_rows:
        if r is None or np.isnan(r['sharpe']) or np.isnan(r['sharpe_se']):
            continue
        lo = r['sharpe'] - z * r['sharpe_se']
        hi = r['sharpe'] + z * r['sharpe_se']
        if lo <= 0 <= hi:
            verdict = "0 DANS l'IC -> edge non distinguable du bruit"
        elif lo > 0:
            verdict = "IC entierement positif (edge OOS plausible)"
        else:  # hi < 0
            verdict = "IC entierement negatif (strategie perdante)"
        print(f"{r['actif']:16} Sharpe={r['sharpe']:+.2f}  SE={r['sharpe_se']:.2f}"
              f"  IC95=[{lo:+.2f}, {hi:+.2f}]  -> {verdict}")


# =====================================================================
#  5) RUN
# =====================================================================

if __name__ == "__main__":
    series = data_loader.load_ticker_series()

    # --- Strategie FIGEE : on injecte le momentum ; defauts du module = params IS ---
    backtester.strategy = strategy_momentum
    p = strategy_momentum
    print("=== PHASE 6 — OUT-OF-SAMPLE SCELLE (passage unique) ===")
    print("Strategie figee : momentum faible rotation")
    print(f"Parametres (figes en IS, NON retouches) : window={p.MOM_WINDOW}, "
          f"k_entry={p.MOM_K_ENTRY}, k_exit={p.MOM_K_EXIT}, "
          f"stop={p.MOM_STOP}, gamma={p.MOM_GAMMA}")

    # --- OOS UNIQUEMENT : start=OOS_START, aucune borne de fin (-> fin des donnees) ---
    res = backtester.run_backtest(series, start=OOS_START)
    if res.empty:
        raise SystemExit("Aucune donnee OOS. Verifiez config.OOS_START et les .pkl.")

    d0, d1 = res["Date"].min().date(), res["Date"].max().date()
    print(f"Fenetre OOS      : {d0} -> {d1}   "
          f"({res['Date'].nunique()} jours de bourse)\n")

    # --- Matrice de performance (format slides Bloch + colonne Gross) ---
    assets = per_asset_metrics(res)
    port_avec = portfolio_metrics(res, "PORTF (avec RUT)", exclude=())
    port_sans = portfolio_metrics(res, "PORTF (sans RUT)", exclude=("RUT",))

    print("===== MATRICE DE PERFORMANCE OOS (Sharpe & MaxDD calcules sur le NET) =====")
    print_matrix(assets, [port_avec, port_sans])

    # --- Ce qui tranche la conclusion : l'IC du Sharpe sur ~40 jours ---
    print_sharpe_ci([port_avec, port_sans])

    print("\nRappel : ceci est le SEUL passage OOS.")
