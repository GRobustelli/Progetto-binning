#%% (1) student depression
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from requests.packages import target

import support_functions as sf
import config as c
from funzioni_confronto import confronto_metodi_tabella
load_dotenv()

#%% 2 student depression
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


#%% 3 adult

info_adult = c.DATASETS_CONFIG["adult"]

dataset_adult = sf.load_dataset(c.ADULT_DATASET,
                                has_header = False,
                                col_name=info_adult["colonne"])

dataset_adult_clean = sf.opt_clean_dataset(dataset_adult,
                                           info_adult["tipo"],
                                           drop_feature_na = True,
                                           target = info_adult["target"])

X_ad = dataset_adult_clean.drop(columns = info_adult["target"]).copy()
Y_ad = dataset_adult_clean[info_adult["target"]].copy()

tabella_confronto_adult = confronto_metodi_tabella(X_ad,Y_ad,c.DEFAULT_N_BINS)
print(tabella_confronto_adult.to_string(index=False))

#%% 4 irrigatio

info_irrigation = c.DATASETS_CONFIG["irrigation"]
dataset_irrigation = sf.load_dataset(c.IRRIGATION_DATASET, has_header = True)

dataset_irrigation_clean = sf.opt_clean_dataset(dataset_irrigation,
                                                info_irrigation["tipo"],
                                                drop_feature_na = True,
                                                target = info_irrigation["target"]
                                                )
X_ir = dataset_irrigation_clean.drop(columns = [info_irrigation["target"]]).copy()
Y_ir = dataset_irrigation_clean[info_irrigation["target"]].copy()

tabella_confronto_irrigation = confronto_metodi_tabella(X_ir,Y_ir,c.DEFAULT_N_BINS)
print(tabella_confronto_irrigation.to_string(index=False))

#%% 5 diabete 3 classi non bilanciato

info_db_3_classi = c.DATASETS_CONFIG["diabete_3"]

dataset_db_3_classi = sf.load_dataset(c.DIABETE_3_CLASSI, has_header = True)
dataset_db_3_classi_clean = sf.opt_clean_dataset(dataset_db_3_classi,
                                                 info_db_3_classi["tipo"],
                                                 drop_feature_na = True,
                                                 target = info_db_3_classi["target"])

X_db3 = dataset_db_3_classi_clean.drop(columns = [info_db_3_classi["target"]]).copy()
Y_db3 = dataset_db_3_classi_clean[info_db_3_classi["target"]].copy()

tabella_confronto_db3 = confronto_metodi_tabella(X_db3,Y_db3,c.DEFAULT_N_BINS)
print(tabella_confronto_db3.to_string(index=False))

#%% 6 diabete 2 classi non bilanciato

info_db2_nb = c.DATASETS_CONFIG["diabete_2_nb"]

dataset_db2_nb = sf.load_dataset(c.DIABETE_3_CLASSI, has_header = True)
dataset_db2_nb_clean = sf.opt_clean_dataset(dataset_db2_nb,
                                            info_db2_nb["tipo"],
                                            drop_feature_na = True,
                                            target=info_db2_nb["target"])

X_db2_nb = dataset_db2_nb_clean.drop(columns = [info_db2_nb["target"]]).copy()
Y_db2_nb = dataset_db2_nb_clean[info_db2_nb["target"]].copy()

tabella_confronto_db2_nb = confronto_metodi_tabella(X_db2_nb,Y_db2_nb,c.DEFAULT_N_BINS)

#%% 7 diabete 2 classi bilanciato

info_db2_bl = c.DATASETS_CONFIG["diabete_2_bl"]
dataset_db2_bl = sf.load_dataset(c.DIABETE_3_CLASSI, has_header = True)
dataset_db2_bl_clean = sf.opt_clean_dataset(dataset_db2_bl,
                                            info_db2_bl["tipo"],
                                            drop_feature_na = True,
                                            target=info_db2_bl["target"])
X_db2_bl = dataset_db2_bl_clean.drop(columns = [info_db2_bl["target"]]).copy()
Y_db2_bl = dataset_db2_bl_clean[info_db2_bl["target"]].copy()

tabella_confronto_db2_bl = confronto_metodi_tabella(X_db2_bl,Y_db2_bl,c.DEFAULT_N_BINS)
print(tabella_confronto_db2_bl.to_string(index=False))
