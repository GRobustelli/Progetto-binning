#%%
import pandas as pd
from optbinning import BinningProcess
import support_functions as sf
from optbinning import MulticlassOptimalBinning
import config as c
"""
Funzione che utilizzo per adesso per applicare optbinning sui miei dataset
:param datapath: è il path dove si trova il dataset 
:param dataset_name: è il nome del dataset
:param sep: specifica quale valore viene utilizzato come separatore (, ; ecc.)
:param has_header: specifica se il dataset ha un header 
:param col_name: il nome delle colonne 
:param na_values: specifica quale valore viene utilizzato per specificare i valori mancanti
:return: df_clean:dataframe =  dataset clean prima di essere binnato
         df_ft:dataframe = dataset dopo essere stato binnanto (forse perché opt binning non accetta l'intero dataset)       
"""

max_n_bins = c.DEFAULT_N_BINS
from optbinning import OptimalBinning

def opt_binning(datapath, dataset_name,
                sep: str = ",",
                has_header: bool = False,
                col_name: list = None,
                na_values = None,
                target: str = None,
                is_multiclass: bool = False):

    df = sf.load_dataset(datapath,
                         sep=sep,
                         has_header=has_header,
                         col_name=col_name,
                         na_values=na_values)

    if target is not None:
        df_clean = sf.opt_clean_dataset(df,
                                        dataset_name,
                                        target=target,
                                        drop_feature_na=True)
    else:
        raise ValueError("Colonna target mancante, va specificata")

    X = df_clean.drop(columns=[target]).copy()
    y = df_clean[target].copy()

    variables = X.columns

    print(f"columns : {variables}")

    X_tf = pd.DataFrame(index=X.index)

    #Nel caso volessi passare anche il modello
    #binning_models = {}

    if is_multiclass:
        for var in variables:
            optb = MulticlassOptimalBinning(name = var,
                                            solver = "cp",
                                            max_n_bins=max_n_bins)
            optb.fit(X[var], y)

            print("optimalbinning status = ", optb.status)

            print(f"\n--- split per la colonna {var} --- \n"
                  f"Numero effettivo: {len(optb.splits) + 1}\n"
                  f"Soglie: {optb.splits}\n")
            X_tf[var] = optb.transform(X[var].values, metric="indices")

    else:
        for var in variables:
            optb = OptimalBinning(name=var,
                                  dtype="numerical",
                                  solver="cp",
                                  max_n_bins=max_n_bins)
            optb.fit(X[var], y)

            print("optimalbinning status = ", optb.status)

            print(f"\n--- split per la colonna {var} --- \n"
                  f"Numero effettivo: {len(optb.splits) + 1}\n"
                  f"Soglie: {optb.splits}\n")
            X_tf[var] = optb.transform(X[var].values, metric="indices")

    return X, X_tf


