# %%  1 setup
import numpy as np
import pandas as pd
import support_functions as sf
import binning_kbinsdiscretizer as bk
import binning_opt_binning as ob
datapath_student_depression = "C:\\datasets\\student_depression_dataset.csv"
student_depression = "student_depression"

datapath_adult = "C:\\datasets\\adult.data"
adult = "adult"
colonne_adult = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "Salary"
]

datapath_irrigation = "C:\\datasets\\irrigation_prediction.csv"
irrigation = "irrigation"

"""
#Dataset NON BILANCIATO ha 21 features variables, ha 3 classi per la variabile target Diabates_binary:
    classe 0: no diabete o durante gravidanza = 0
    classe 2: pre-diabete = 1
    classe 3: diabete = 2
"""
datapath_diabetes_health_indicator = "C:\\datasets\\diabete\\diabetes_012_health_indicators_BRFSS2015.csv"
diabetes_health = "diabetes_health"


"""
#Dataset BILANCIATO ha 21 features variables, ha 2 classi per la variabile target Diabates_binary:
    Classe 1: no diabete = 0
    Classe 2: pre-diabete, diabete = 1
"""
datapath_diabetes_5050_bilanciato = "C:\\datasets\\diabete\\diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
diabetes_5050_bilanciato = "diabetes_health_5050"

"""
#Dataset NON BILANCIATO ha 21 feature variables, ha 2 classi per la variabile target Diabetes_binary:
    Classe 1: no diabetes = 0
    Classe 2: pre-diabete, diabete = 1
"""
datapath_diabetes_5050_non_bilanciato = "C:\\datasets\\diabete\\diabetes_binary_health_indicators_BRFSS2015.csv"
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
target_col = "Depression"

student_clean_ob, student_tf_ob = ob.opt_binning(datapath_student_depression,
                                           student_depression,
                                           has_header=True,
                                           target=target_col)

print(student_clean_ob[:10])
print(student_tf_ob[:10])
