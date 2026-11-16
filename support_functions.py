import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def info_dataset(datapath):
    df = pd.read_csv(datapath)
    df.info()


def load_dataset(filepath: str,
                 #target_col: str,   per adesso non mi serve prepararli per un modello quindi la colonna target la ignoro per adesso
                 sep: str = ",",
                 has_header: bool = False,
                 col_name: list = None,
                 na_values=None,
                ):
    if na_values is None:
        na_values = ["?", ",", ";", "nan", ""]

    header_arg = 0 if has_header else None
    ds = pd.read_csv(filepath,
                     sep=sep,
                     header = header_arg,
                     names = col_name if not has_header else None,
                     na_values=na_values,
                     skipinitialspace=True)

    return ds

def clean_dataset(dataset, tipo):
    if tipo == "student_depression":
        colonne_continue = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours',
                            'Financial Stress']

        df = dataset[colonne_continue]
        df = df.dropna().reset_index(drop=True)
        return df

    elif tipo == "adult":
        colonne_continue = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        df = dataset[colonne_continue].copy()
        df = df.dropna().reset_index(drop=True)
        return df

    elif tipo == "irrigation":
        df = dataset.select_dtypes(include="number")
        df = df.dropna().reset_index(drop=True)
        return df

    elif tipo == "diabetes":
        #Il binning mi risulta inutile la feature presenta solo valore 0 o 1, seleziono per adesso
        #solo le colonne con valori interessanti
        colonne_continue = ["BMI","MentHlth","PhysHlth","Age","Education", "Income"]
        df = dataset[colonne_continue].copy()
        df = df.dropna().reset_index(drop=True)
        return df

    return None


def plot_binning_comparison(X_before, X_after, column, strategy_name="Quantile"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 1. PRIMA: Istogramma della variabile originale
    sns.histplot(X_before[column], kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title(f"{column} - Originale")
    axes[0].set_xlabel("Valore originale")
    axes[0].set_ylabel("Frequenza")

    # 2. DOPO: Conteggio per ciascun bin creato
    # Ordiniamo gli indici dei bin (0, 1, 2, ...) sull'asse X
    bin_counts = X_after[column].value_counts().sort_index()
    sns.barplot(x=bin_counts.index.astype(int), y=bin_counts.values, ax=axes[1], color="darkorange")
    axes[1].set_title(f"{column} - Dopo Binning ({strategy_name})")
    axes[1].set_xlabel("Indice del Bin")
    axes[1].set_ylabel("Numero di osservazioni")

    plt.tight_layout()
    plt.show()

def stampa_range_bin(est, nomi_colonne):
    """
    Stampa gli intervalli [min, max) di ciascun bin creato da KBinsDiscretizer.
    """
    for col, edges in zip(nomi_colonne, est.bin_edges_):
        print(f"\n--- Colonna: {col} ({len(edges) - 1} bin effettivi) ---")
        for i in range(len(edges) - 1):
            inf = edges[i]
            sup = edges[i + 1]
            chiusura = "]" if i == len(edges) - 2 else ")"
            print(f"  Bin {i}: [{inf:.3f}, {sup:.3f}{chiusura}")

