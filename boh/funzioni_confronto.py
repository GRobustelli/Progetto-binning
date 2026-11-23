import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
from optbinning import OptimalBinning, MulticlassOptimalBinning

def confronto_metodi_tabella(X: pd.DataFrame, y: pd.Series, n_bins: int = 7)->pd.DataFrame:
    risultati = []
    is_multiclass = y.nunique() > 2

    for col in X.columns:
        x_col = X[[col]].values
        x_1d = X[col].values

        #1. EqualFrequency kbinsdiscretizer
        kbd_q = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        bins_q = kbd_q.fit_transform(x_col).ravel().astype(int)
        splits_q = np.round(kbd_q.bin_edges_[0][1:-1], 2).tolist()

        #2 kmeans kbinsdiscretizer
        kbd_km = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
        bins_km = kbd_km.fit_transform(x_col).ravel().astype(int)
        splits_km = np.round(kbd_km.bin_edges_[0][1:-1], 2).tolist()

        #3 optimalbinning supervisionato
        if is_multiclass:
            optb = MulticlassOptimalBinning(name=col,max_n_bins=n_bins,dtype="numerical")
        else:
            optb = OptimalBinning(name=col,dtype="numerical", max_n_bins=n_bins)

        optb.fit(x_1d,y.values)
        bins_opt = optb.transform(x_1d, metric="indices").astype(int)
        splits_opt = np.round(optb.splits, 2).tolist()

        #Raccolta metriche
        metodi = [
            ("KBinsDiscretizer - equal frequency (quantile)", bins_q, splits_q),
            ("KbinsDiscretizer - k-means (kmeans)", bins_km, splits_km),
            ("OptimalBinning - approccio supervisionato", bins_opt, splits_opt)
        ]

        for metodo, bin_array, split in metodi:
            bin_effettivi = len(np.unique(bin_array))
            mi = mutual_info_score(y.values, bin_array)

            risultati.append({
                "Feature": col,
                "Metodo": metodo,
                "Bin richiesti": n_bins,
                "Bin effettivi": bin_effettivi,
                "Mutual information VS Target": mi,
                "Splits": str(split),
            }
            )

    return pd.DataFrame(risultati)