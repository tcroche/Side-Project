# strategy_momentum.py
import numpy as np
import pandas as pd

import config

# --- Parametres FIGES de la strategie
MOM_WINDOW  = 120     # fenetre lente -> z-score lisse, peu de bascules (faible rotation)
MOM_K_ENTRY = 3.0     # entree seulement si |z| depasse ce seuil -> LEVIER de rotation
MOM_K_EXIT  = 0.3     # sortie quand z revient pres de la moyenne (|z| < k_exit)
MOM_STOP    = 0.005   # stop-loss 0.5% (sur la direction REELLE de la position)
MOM_GAMMA   = 0.3     # raideur du sizing : tanh(0.3*3)=0.72 a l'entree, ~0.95 pour z=5


def rolling_zscore(close, window):
    """z_t = (P_t - moyenne) / ecart-type sur les 'window' dernieres minutes.
    .rolling() est causal : a l'instant t il n'utilise que le passe."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / std


def compute_positions(close, window=MOM_WINDOW, k_entry=MOM_K_ENTRY,
                      k_exit=MOM_K_EXIT, stop_loss=MOM_STOP, gamma=MOM_GAMMA):
    """Positions continues dans [-1, +1] pour UNE journee, indexees comme 'close'.
    0 = flat ; signe = sens (momentum) ; |valeur| = taille (tanh, figee a l'entree).
    Boucle explicite -> causalite et faible rotation evidentes (a defendre)."""
    prices = close.values
    n = len(prices)
    pos = np.zeros(n)

    roll_mean = close.rolling(window).mean().values
    roll_std = close.rolling(window).std().values

    current = 0           # signe de la position courante : -1, 0, +1
    size = 0.0            # taille effective detenue (signee), dans [-1, +1]
    entry_price = np.nan  # prix d'entree (pour le stop-loss)

    for t in range(n):
        # 1) Flat force a la derniere barre -> pas de position overnight
        if t == n - 1:
            pos[t] = 0.0
            break
        # 2) Fenetre pas encore remplie / ecart-type nul -> rester flat
        if np.isnan(roll_mean[t]) or not roll_std[t] > 0:
            pos[t] = 0.0
            current, size = 0, 0.0
            continue

        z = (prices[t] - roll_mean[t]) / roll_std[t]

        if current == 0:
            # 3) ENTREE dans le sens de l'ecart (momentum / continuation).
            #    La taille est FIGEE ici : w = tanh(gamma*z) -> signe + magnitude.
            if z > k_entry:
                current, entry_price = +1, prices[t]
                size = float(np.tanh(gamma * z))           # > 0 (long)
            elif z < -k_entry:
                current, entry_price = -1, prices[t]
                size = float(np.tanh(gamma * z))           # < 0 (short)
        else:
            # 4) SORTIE : retour vers la moyenne (|z| < k_exit) ou stop-loss touche
            exit_signal = abs(z) < k_exit
            if stop_loss is not None and entry_price > 0:
                ret = current * (prices[t] - entry_price) / entry_price
                if ret < -stop_loss:
                    exit_signal = True
            if exit_signal:
                current, size, entry_price = 0, 0.0, np.nan

        pos[t] = size     # taille MAINTENUE tant qu'on n'a ni tourne ni stoppe

    return pd.Series(pos, index=close.index, name="position")


if __name__ == "__main__":
    import data_loader

    series = data_loader.load_ticker_series()
    tk = "GSPC" if "GSPC" in series else sorted(series)[0]
    df = series[tk]

    first_day = df.index.normalize().min()
    day = df[df.index.normalize() == first_day]
    close = day["Close"]

    pos = compute_positions(close)
    prev = pos.shift(1).fillna(0.0)
    entries = int(((prev == 0) & (pos != 0)).sum())     # passages flat -> en position
    in_pos = pos[pos != 0]

    print(f"Indice teste     : {tk}  ({first_day.date()})")
    print(f"Barres ce jour   : {len(close)}")
    print(f"Position finale  : {pos.iloc[-1]:.3f}   (doit etre 0 -> flat EOD)")
    print(f"Entrees ce jour  : {entries}   (faible rotation attendue : 0-2)")
    print(f"Barres en pos.   : {len(in_pos)}")
    if len(in_pos):
        sizes = sorted({round(abs(v), 3) for v in in_pos.values})
        print(f"Tailles |pos|    : {sizes}   (tanh -> fractionnaire, jamais > 1)")
        print(f"|pos| max / moy  : {in_pos.abs().max():.3f} / {in_pos.abs().mean():.3f}")
    else:
        print("Aucune position ce jour-la (|z| n'a pas franchi le seuil d'entree).")
