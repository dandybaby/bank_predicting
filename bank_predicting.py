import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from collections import defaultdict

df = pd.read_csv("train.csv")
print(df.isnull().sum())
df.drop(["ult_fec_cli_1t", "conyuemp"], axis=1, inplace=True)

df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["antiguedad"] = pd.to_numeric(df["antiguedad"], errors="coerce")
df["indrel_1mes"] = pd.to_numeric(df["indrel_1mes"], errors="coerce")

grouped = df.groupby("nomprov").agg({"renta": lambda x: x.median(skipna=True)}).reset_index()
new_incomes = pd.merge(df, grouped, how="inner", on="nomprov").loc[:, ["nomprov", "renta_y"]]
new_incomes = new_incomes.rename(columns={"renta_y": "renta"})
df.sort_values("nomprov", inplace=True)
df = df.reset_index()
new_incomes = new_incomes.reset_index()

df.loc[df.renta.isnull(), "renta"] = new_incomes.loc[df.renta.isnull(), "renta"].reset_index()
df.loc[df.renta.isnull(), "renta"] = df.loc[df.renta.notnull(), "renta"].median()
df.sort_values(by="fecha_dato", inplace=True)


df = df.dropna(axis=0)
age_mean = df["antiguedad"].mean()
df[df["antiguedad"]<0]=age_mean


df["tot_products"] = df.loc[:,"ind_ahor_fin_ult1":"ind_recibo_ult1"].sum(axis=1)
df["tot_products"]   = pd.to_numeric(df["tot_products"], errors="coerce")
df['pais_residencia'].describe()
df.drop("pais_residencia", axis=1, inplace = True)
df['ind_empleado'].value_counts()
df.drop("ind_empleado",axis=1,inplace=True)
df.antiguedad = pd.to_numeric(df.antiguedad,errors="coerce")
df.ind_nomina_ult1 = pd.to_numeric(df.ind_nomina_ult1)
df.ind_nom_pens_ult1 = pd.to_numeric(df.ind_nom_pens_ult1)
df.isnull().values.any()

df_train = df.drop_duplicates(['ncodpers'], keep='last')
df_train.fillna(0, inplace=True)
unique_ids   = pd.Series(df_train["ncodpers"].unique())
sample = pd.read_csv('sample_submission.csv')
limit_people = 10000
unique_id    = unique_ids.sample(n=limit_people)
models = {}
id_preds = defaultdict(list)
ids = df_train['ncodpers'].values
#Delete in final model (Done before)
feature_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
for col in feature_cols:
    df[col] = df[col].astype(int)
for c in df_train.columns:
    if c != 'ncodpers':
        #print(c)
        y_train = df_train[c]
        x_train = df_train.drop([c, 'ncodpers'], 1)


clf = LogisticRegression()
clf.fit(x_train, y_train)
p_train = clf.predict_proba(x_train)[:,1]

models[c] = clf
for id, p in zip(ids, p_train):
    id_preds[id].append(p)

print(roc_auc_score(y_train, p_train))