import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def info_dataset(datapath):
    df = pd.read_csv(datapath)
    df.info()


def load_dataset(datapath):
    df = pd.read_csv(datapath)

    return df

def clean_dataset(dataset, tipo):
    if tipo == "student_depression":
        colonne_continue = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours',
                            'Financial Stress']
        df = dataset
        X = df[colonne_continue]
        X_clean = X.dropna()
        return X_clean

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
