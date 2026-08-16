# compare_strategies.py
import datetime as dt

import config
import data_loader
from m2_backtester import backtester, portfolio

import strategy                      # 1) mean-reversion ( defauts config)
import strategy_momentum             # 2) momentum faible rotation (figee)
import strategy_intraday_momentum    # 3) intraday momentum Gao et al. (a priori)

OOS_START = dt.date.fromisoformat(config.OOS_START)
IS_END = OOS_START - dt.timedelta(days=1)

STRATS = [
    ("mean-reversion", strategy),
    ("momentum faible rot.",  strategy_momentum),
    ("intraday-mom",    strategy_intraday_momentum),
]


def run_is(stratmod, series):
    """Backtest IS d'une strategie -> (res long, matrice Date x Ticker netRet)."""
    backtester.strategy = stratmod                     
    res = backtester.run_backtest(series, end=IS_END)  # IN-SAMPLE uniquement
    ret = portfolio.returns_matrix(res, "netRet")
    return res, ret


def per_asset_table(res, ret, label):
    print(f"\n--- {label} : par actif (IS) ---")
    print(f"{'Ticker':8}{'netRet%':>10}{'netSharpe':>11}{'tr/j':>7}")
    trj = res.groupby("Ticker")["numTrade"].mean()
    for tk in ret.columns:
        col = ret[tk].dropna()
        print(f"{tk:8}{100 * col.sum():>10.2f}"
              f"{portfolio.ann_sharpe(col):>11.2f}{trj.get(tk, float('nan')):>7.1f}")


def portfolio_line(ret, exclude=()):
    cols = [c for c in ret.columns if c not in exclude]
    sub = ret[cols]
    p = portfolio.portfolio_returns(sub, portfolio.equal_weights(cols))
    return 100 * p.sum(), portfolio.ann_sharpe(p), 100 * portfolio.max_drawdown(p)


if __name__ == "__main__":
    series = data_loader.load_ticker_series()

    # 1) Detail par actif pour chaque strategie
    cache = {}
    for label, mod in STRATS:
        res, ret = run_is(mod, series)
        cache[label] = ret
        per_asset_table(res, ret, label)

    print(f"\n(IS : {cache[STRATS[0][0]].index.min().date()} -> "
          f"{cache[STRATS[0][0]].index.max().date()})")

    # 2) Synthese portefeuille equal-weight
    for title, exc in [("AVEC RUT", ()), ("SANS RUT", ("RUT",))]:
        print(f"\n===== PORTEFEUILLE equal-weight — {title} (IS) =====")
        print(f"{'strategie':24}{'netRet%':>10}{'Sharpe':>9}{'maxDD%':>9}")
        for label, _ in STRATS:
            r, s, dd = portfolio_line(cache[label], exclude=exc)
            print(f"{label:24}{r:>10.2f}{s:>9.2f}{dd:>9.2f}")
