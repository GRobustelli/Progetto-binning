#%% (1) student depression
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from requests.packages import target

import support_functions as sf
import config as c
from funzioni_confronto import confronto_metodi_tabella
load_dotenv()
info_student = c.DATASETS_CONFIG["depression"]
dataset_student = sf.load_dataset(c.DEPRESSION_DATASET,
                                  has_header = True)

dataset_student_clean = sf.opt_clean_dataset(dataset_student,
                                             info_student["tipo"],
                                             drop_feature_na=True,
                                             target = info_student["target"])

X = dataset_student_clean.drop(columns = [info_student["target"]]).copy()
Y = dataset_student_clean[info_student["target"]].copy()

tabella_confronto_student = confronto_metodi_tabella(X,Y,c.DEFAULT_N_BINS)

print(tabella_confronto_student.to_string(index=False))


