import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import support_functions as sf

colonne_continue_adult = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

colonne_continue_student = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours',
                    'Financial Stress']


def kbins_discretizer_strategy_quantile(datapath, dataset_name,
                                        sep: str = ",",
                                        has_header: bool = False,
                                        col_name: list = None,
                                        na_values = None,
                                        ):

    """
    Per adesso carico il file csv e scelgo solo le colonne con valori numerici
    Devo chiedere se trasformare le altre informazioni in valori numerici per poter fare
    """
    df = sf.load_dataset(datapath,
                         sep=sep,
                         has_header=has_header,
                         col_name=col_name,
                         na_values=na_values)
    """
     KBDiscretizer non accetta valori NaN nel metodo non supervisionato, quindi modifico
     Il mio dataset per adesso eliminando le righe che presentano valori NaN
    """
    X_clean = sf.clean_dataset(df, dataset_name)

    # Utilizzo KBinsDiscretizer con strategia "quantile" e n_bins=7 come consigliati

    est = KBinsDiscretizer(n_bins=7, encode="ordinal", strategy="quantile")
    est.set_output(transform="pandas")
    X_tf = est.fit_transform(X_clean)

    # Dato che dataset come studendt depression ha molte colonne con valori discreti
    # qui stampo quanti bin effettivamente kbinsdiscretizer ha effettivamente creato

    print(f'Per il dataset {dataset_name}, utilizzando la strategia quantile:\n')
    for col, n in zip(X_clean.columns, est.n_bins_):
        print(f"{col}: {n} bin effettivi")

    sf.stampa_range_bin(est,X_clean.columns)

    return X_clean, X_tf

    """
    # Dato che questo dataset ha molti valori che vanno da 1-5 provo anche con n_bins = 5
    est2 = KBinsDiscretizer(n_bins=5,encode="ordinal" ,strategy="quantile")
    est2.fit(X_clean2)
    X_tf2 = est2.transform(X_clean2)
    """

def kbins_discretizer_strategy_kmeans(datapath,
                                      dataset_name,
                                      sep: str = ",",
                                      has_header: bool = False,
                                      col_name: list = None,
                                      na_values = None,
                                      ):
    ds = sf.load_dataset(datapath,
                         sep=sep,
                         has_header=has_header,
                         col_name=col_name,
                         na_values=na_values)
    ds_clean = sf.clean_dataset(ds, dataset_name)

    if ds_clean is not None:
        est = KBinsDiscretizer(n_bins=7, encode="ordinal", strategy="kmeans", random_state=42)
        est.set_output(transform="pandas")
        ds_tf = est.fit_transform(ds_clean)
    else:
        raise ValueError("dataset caricato non riconosciuto")

    print(f'Per il dataset {dataset_name}, utilizzando la strategia kmeans:\n')
    for col, n in zip(ds_clean.columns, est.n_bins_):
        print(f"{col}: {n} bin effettivi")

    print("**************************************************************************")
    sf.stampa_range_bin(est,ds_clean.columns)

    return ds_clean, ds_tf





