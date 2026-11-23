# %%  1 setup
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import support_functions as sf
import binning_kbinsdiscretizer as bk
import binning_opt_binning as ob
import config as c
"""
load_dotenv()

def get_path(env_var: str) -> Path:
    val = os.getenv(env_var)
    if not val:
        raise EnvironmentError(f"Variabile '{env_var}' non trovata nel file .env!")
    return Path(val)
"""


datapath_student_depression = c.DEPRESSION_DATASET
student_depression = "student_depression"

datapath_adult = c.ADULT_DATASET
adult = "adult"
colonne_adult = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "Salary"]


datapath_irrigation = c.IRRIGATION_DATASET
irrigation = "irrigation"

"""
#Dataset NON BILANCIATO ha 21 features variables, ha 3 classi per la variabile target Diabates_binary:
    classe 0: no diabete o durante gravidanza = 0
    classe 2: pre-diabete = 1
    classe 3: diabete = 2
"""

datapath_diabetes_health_indicator = c.DIABETE_3_CLASSI
diabetes_health = "diabetes_health"


"""
#Dataset BILANCIATO ha 21 features variables, ha 2 classi per la variabile target Diabates_binary:
    Classe 1: no diabete = 0
    Classe 2: pre-diabete, diabete = 1
"""
datapath_diabetes_5050_bilanciato = c.DIABETE_2_CLASSI_BILANCIATO
diabetes_5050_bilanciato = "diabetes_health_5050"

"""
#Dataset NON BILANCIATO ha 21 feature variables, ha 2 classi per la variabile target Diabetes_binary:
    Classe 1: no diabetes = 0
    Classe 2: pre-diabete, diabete = 1
"""
datapath_diabetes_5050_non_bilanciato = c.DIABETE_2_CLASSI_NONBILANCIATO
diabetes_5050_non_balance = "diabetes_5050_non_balance"
diabete_generic = "diabetes"


# %% 2 Student

sf.info_dataset(datapath_student_depression)
student_clean, student_tf = bk.kbins_discretizer_strategy_quantile(datapath_student_depression,
                                                                   student_depression,
                                                                   has_header = True)

student_clean_km, student_tf_km = bk.kbins_discretizer_strategy_kmeans(datapath_student_depression,
                                                                       student_depression,
                                                                       has_header = True)

#%% 3 adult
adult_clean, adult_tf = bk.kbins_discretizer_strategy_quantile(datapath_adult,
                                                               adult,
                                                               has_header=False,
                                                               sep=",",
                                                               col_name = colonne_adult,
                                                               na_values='?'
                                                               )

adult_clean_km, adult_tf_km = bk.kbins_discretizer_strategy_kmeans(datapath_adult,
                                                                   adult,
                                                                   has_header=False,
                                                                   col_name=colonne_adult,
                                                                   na_values='?')

#%% 4 irrigation
irrigation_clean, irrigation_tf = bk.kbins_discretizer_strategy_quantile(datapath_irrigation,
                                                                         irrigation,
                                                                         has_header=True)

irrigation_clean_km, irrigation_tf_km = bk.kbins_discretizer_strategy_kmeans(datapath_irrigation,
                                                                             irrigation,
                                                                             has_header=True)

#%% 5 diabete 3 classi non bilanciato

sf.info_dataset(datapath_diabetes_health_indicator)

diabetes_1_clean, diabetes_1_tf = bk.kbins_discretizer_strategy_quantile(datapath_diabetes_health_indicator,
                                                                        diabete_generic,
                                                                        has_header=True)

diabetes_1_clean_km, diabetes_1_tf_km = bk.kbins_discretizer_strategy_kmeans(datapath_diabetes_health_indicator,
                                                                             diabete_generic,
                                                                             has_header=True)
print("Database pre e post fit e transform strategia QUANTILE \n")
print(diabetes_1_clean[:10])
print(diabetes_1_tf[:10])

print("\n****************************************************************\n")

print("Database pre e post fit e transform strategia K-means \n")
print(diabetes_1_clean_km[:10])
print(diabetes_1_tf_km[:10])

#%% 6 diabete 2 classi bilanciato

sf.info_dataset(datapath_diabetes_5050_bilanciato)

diabetes_2_clean, diabetes_2_tf = bk.kbins_discretizer_strategy_quantile(datapath_diabetes_5050_bilanciato,
                                                                         diabete_generic,
                                                                         has_header=True)
diabetes_2_clean_km, diabetes_2_tf_km = bk.kbins_discretizer_strategy_kmeans(datapath_diabetes_5050_bilanciato,
                                                                         diabete_generic,
                                                                         has_header=True)

print(f"Database {diabetes_5050_bilanciato} pre e post fit e transform strategia QUANTILE \n")
print(diabetes_2_clean[:10])
print(diabetes_2_tf[:10])

print("\n****************************************************************\n")

print(f"Database {diabetes_5050_bilanciato} pre e post fit e transform strategia K-means \n")
print(diabetes_2_clean_km[:10])
print(diabetes_2_tf_km[:10])


#%% 7 diabete 2 classi non bilanciato

sf.info_dataset(datapath_diabetes_5050_bilanciato)

diabetes_3_clean, diabetes_3_tf = bk.kbins_discretizer_strategy_quantile(datapath_diabetes_5050_non_bilanciato,
                                                                         diabete_generic,
                                                                         has_header=True)
diabetes_3_clean_km, diabetes_3_tf_km = bk.kbins_discretizer_strategy_kmeans(datapath_diabetes_5050_non_bilanciato,
                                                                         diabete_generic,
                                                                         has_header=True)

print(f"Database {diabetes_5050_non_balance} pre e post fit e transform strategia QUANTILE \n")
print(diabetes_3_clean[:10])
print(diabetes_3_tf[:10])

print("\n****************************************************************\n")

print(f"Database {diabetes_5050_non_balance} pre e post fit e transform strategia K-means \n")
print(diabetes_3_clean_km[:10])
print(diabetes_3_tf_km[:10])


#%% 8 INIZIA QUI OPTBINNING - student depression

sf.info_dataset(datapath_student_depression)
target_col_student = "Depression"

student_clean_ob, student_tf_ob = ob.opt_binning(datapath_student_depression,
                                           student_depression,
                                           has_header=True,
                                           target=target_col_student)

print(student_clean_ob[:10])
print(student_tf_ob[:10])

#%% 9 adult

sf.info_dataset(datapath_adult)
target_col_adult = "Salary"

adult_clean_ob, adult_tf_ob = ob.opt_binning(datapath_adult,
                                               adult,
                                               sep = ",",
                                               has_header=False,
                                               col_name=colonne_adult,
                                               na_values='?',
                                               target = target_col_adult,
                                               is_multiclass = False
                                               )
adult_clean_ob.info()
print(adult_clean_ob[:10])
print(adult_tf_ob[:10])

#%% 10 irrigation

sf.info_dataset(datapath_irrigation)
target_col_irrigation = "Irrigation_Need"

irrigation_clean_ob, irrigation_tf_ob = ob.opt_binning(datapath_irrigation,
                                                       irrigation,
                                                       has_header=True,
                                                       is_multiclass=True,
                                                       target = target_col_irrigation)
print(irrigation_clean_ob[:10])
print(irrigation_tf_ob[:10])

#%% 11 diabete 3 classi non bilanciato

sf.info_dataset(datapath_diabetes_health_indicator)

target_col_diabetes_3class = "Diabetes_012"

diabetes_1_clean_ob, diabetes_1_tf_ob = ob.opt_binning(datapath_diabetes_health_indicator,
                                                       diabete_generic,
                                                       has_header=True,
                                                       is_multiclass=True,
                                                       target = target_col_diabetes_3class)
print(diabetes_1_clean_ob[:10])
print(diabetes_1_tf_ob[:10])

#%% 12 diabete 2 classi bilanciato

sf.info_dataset(datapath_diabetes_5050_bilanciato)
target_col_diabetes_2class = "Diabetes_binary"

diabetes_2_clean_ob, diabetes_2_tf_ob = ob.opt_binning(datapath_diabetes_5050_bilanciato,
                                                       diabete_generic,
                                                       has_header=True,
                                                       is_multiclass=False,
                                                       target = target_col_diabetes_2class)
print(diabetes_2_clean_ob[:10])
print(diabetes_2_tf_ob[:10])

#%% 13 diabete 2 classi non bilanciato

sf.info_dataset(datapath_diabetes_5050_non_bilanciato)
target_col_diabetes_2class = "Diabetes_binary"

diabetes_3_clean_ob, diabetes_3_tf_ob = ob.opt_binning(datapath_diabetes_5050_non_bilanciato,
                                                       diabete_generic,
                                                       has_header=True,
                                                       is_multiclass=False,
                                                       target = target_col_diabetes_2class)
print(diabetes_3_clean_ob[:10])
print(diabetes_3_tf_ob[:10])

