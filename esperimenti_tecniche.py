import numpy as np
import pandas as pd
import support_functions as sf
import binning_kbinsdiscretizer as bk

datapath_student_depression = "C:\\datasets\\student_depression_dataset.csv"
student_depression = "student_depression"
datapath_adult = "C:\\datasets\\adult.data"
adult = "adult"

colonne_adult = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "Salary"
]


student_clean, student_tf = bk.kbins_discretizer_strategy_quantile(datapath_student_depression,
                                                                   student_depression,
                                                                   has_header = True)

student_clean_km, student_tf_km = bk.kbins_discretizer_strategy_kmeans(datapath_student_depression,
                                                                       student_depression,
                                                                       has_header = True)

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

