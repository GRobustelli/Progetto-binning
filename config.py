import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Qui trovo e carico automaticamente .env dalla stessa cartella di config.py
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

def get_path(env_var: str) -> Path:
    val = os.getenv(env_var)
    if not val:
        raise EnvironmentError(f"Variabile '{env_var}' non trovata in config.env!")
    return Path(val)

# --- PERCORSI PRIVATI (letti da config.env) ---
DEPRESSION_DATASET = get_path("DEPRESSION_DATASET")
ADULT_DATASET = get_path("ADULT_DATASET")
IRRIGATION_DATASET = get_path("IRRIGATION_DATASET")
DIABETE_3_CLASSI = get_path("DIABETE_3_CLASSI")
DIABETE_2_CLASSI_NONBILANCIATO = get_path("DIABETE_2_CLASSI_NONBILANCIATO")
DIABETE_2_CLASSI_BILANCIATO = get_path("DIABETE_2_CLASSI_BILANCIATO")

# --- PARAMETRI GLOBALI CONDIVISI ---
DEFAULT_N_BINS = 7
MIN_BIN_SIZE = 0.05
OUTPUT_DIR = BASE_DIR / "risultati"

# Dizionario di configurazione per ogni dataset
DATASETS_CONFIG = {
    "depression": {
        "name": "student_depression",
        "path": DEPRESSION_DATASET,
        "target": "Depression",
        "tipo": "student_depression"
    },
    "adult": {
        "name": "adult",
        "path": ADULT_DATASET,
        "target": "Salary",
        "tipo": "adult"

    },
    "irrigation": {
        "name": "irrigation",
        "path": IRRIGATION_DATASET,
        "target": "Irrigation_Need",
        "tipo": "irrigation"
    },
    "diabete_3": {
        "name": "diabete_3_nonbilanciato",
        "path": DIABETE_3_CLASSI,
        "target": "Diabetes_012",
        "tipo": "diabetes"
    },
    "diabete_2_nb": {
        "name": "diabete_2_nonbilanciato",
        "path": DIABETE_2_CLASSI_NONBILANCIATO,
        "target": "Diabetes_binary",
        "tipo": "diabetes"
    },
    "diabete_2_bl":{
        "name": "diabete_2_bilanciato",
        "path": DIABETE_2_CLASSI_BILANCIATO,
        "target": "Diabetes_binary",
        "tipo": "diabetes"
    }
}