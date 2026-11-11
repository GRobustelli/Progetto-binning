import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

#Per adesso carico il file csv e scelgo solo le colonne con valori numerici
#Devo chiedere se trasformare le altre informazioni in valori numerici per poter fare
#Una discretizzazione dell'intero dataset per esempio gender: male = 0 e female = 1

colonne_continue = ['Age','Academic Pressure','CGPA','Study Satisfaction','Work/Study Hours','Financial Stress']
df = pd.read_csv("C:\\datasets\\student_depression_dataset.csv")
X = df[['Age','Academic Pressure','CGPA','Study Satisfaction','Work/Study Hours','Financial Stress']]

#KBDiscretizer non accetta valori NaN nel metodo non supervisionato, quindi modifico
#Il mio dataset per adesso eliminando le righe che presentano valori NaN

X_clean = X.dropna()
X_clean2 = X_clean.copy()

print("Dataset senza NaN e con colonne con variabili discretizzabili:\n",X_clean)

#Utilizzo KBinsDiscretizer con strategia "quantile" e n_bins=7 come consigliati
est = KBinsDiscretizer(n_bins=7,encode="ordinal" ,strategy="quantile")
est.fit(X_clean)
X_tf = est.transform(X_clean)
print(X_tf)

#Dato che questo dataset ha molti valori che vanno da 1-5 provo anche con n_bins = 5

est2 = KBinsDiscretizer(n_bins=5,encode="ordinal" ,strategy="quantile")
est2.fit(X_clean2)
X_tf2 = est2.transform(X_clean2)





