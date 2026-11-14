import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import support_functions as sf

datapath_student_depression = "C:\\datasets\\student_depression_dataset.csv"
student_depression = "student_depression"

def kbins_discretizer_strategy_quantile():
    # Per adesso carico il file csv e scelgo solo le colonne con valori numerici
    # Devo chiedere se trasformare le altre informazioni in valori numerici per poter fare
    # Una discretizzazione dell'intero dataset per esempio gender: male = 0 e female = 1

    df = sf.load_dataset(datapath_student_depression)

    # KBDiscretizer non accetta valori NaN nel metodo non supervisionato, quindi modifico
    # Il mio dataset per adesso eliminando le righe che presentano valori NaN

    X_clean = sf.clean_dataset(df, student_depression)
    X_clean2 = X_clean.copy()
    print("Dataset senza NaN e con colonne con variabili discretizzabili:\n", X_clean[:10])

    # Utilizzo KBinsDiscretizer con strategia "quantile" e n_bins=7 come consigliati

    est = KBinsDiscretizer(n_bins=7, encode="ordinal", strategy="quantile")
    est.set_output(transform="pandas")
    est.fit(X_clean)
    X_tf = est.transform(X_clean)
    # Dato che studendt depression ha molte colonne con valori discreti
    # qui stampo quanti bin effettivamente kbinsdiscretizer ha effettivamente creato

    print("Per il dataset student depression:\n")
    for col, n in zip(X_clean.columns, est.n_bins_):
        print(f"{col}: {n} bin effettivi")

    sf.plot_binning_comparison(X_clean,X_tf,"Work/Study Hours", "Quantile - 7 bins")
    # Dato che questo dataset ha molti valori che vanno da 1-5 provo anche con n_bins = 5

    """
    est2 = KBinsDiscretizer(n_bins=5,encode="ordinal" ,strategy="quantile")
    est2.fit(X_clean2)
    X_tf2 = est2.transform(X_clean2)
    """

kbins_discretizer_strategy_quantile()




