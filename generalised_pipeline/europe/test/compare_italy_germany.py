
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

df_ger = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\germany\\processed_germany.pkl")
df_ger['date'] = pd.to_datetime(df_ger['date'])
df_ger = df_ger[df_ger.date.dt.year > 2009]
df_ger = df_ger[df_ger.date.dt.month == 1]
df_ita = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\italy\\processed_italy.pkl")
df_ita['date'] = pd.to_datetime(df_ita['date'])
df_ita = df_ita[df_ita.date.dt.year > 2009]
df_ita = df_ita[df_ita.date.dt.month == 1]

countries = list(set(df_ita.origin).union(set(df_ger.origin)))
print("countries in ITA data and not in GER:")
print(set(df_ita.origin) - set(df_ger.origin))
print("=====================================")
print("countries in GER data and not in ITA:")
print(set(df_ger.origin) - set(df_ita.origin))

#########
pbar = tqdm(countries)
for country in pbar:
    pbar.set_description(f"Processing {country} ...")
    for date in df_ita.date.unique():
        for sex in ["male", "female"]:
            try:
                df_ger.loc[(df_ger.origin == country) & (df_ger.date == date) & (df_ger.sex == sex), "pct"] = (
                    df_ger.loc[(df_ger.origin == country) & (df_ger.date == date) & (df_ger.sex == sex), "n_people"] /
                    df_ger.loc[(df_ger.origin == country) & (df_ger.date == date) & (df_ger.sex == sex), "n_people"].sum())
            except:
                print(f"No people from {country} in Germany")
            try:
                df_ita.loc[(df_ita.origin == country) & (df_ita.date == date) & (df_ita.sex == sex), "pct"] = (
                    df_ita.loc[(df_ita.origin == country) & (df_ita.date == date)& (df_ita.sex == sex), "n_people"] /
                    df_ita.loc[(df_ita.origin == country) & (df_ita.date == date)& (df_ita.sex == sex), "n_people"].sum())
            except:
                print(f"No people from {country} in Italy")

df_ger = df_ger[df_ger.date.dt.year <= 2021]

df_mer = (df_ita[['date', 'origin', 'age_group', 'sex', 'n_people', 'pct']].
          merge(df_ger[['date', 'origin', 'age_group', 'sex', 'n_people', 'pct']],
                on = ['date', 'origin', 'age_group', 'sex'], how = 'outer',
                suffixes = ('_ita', '_ger')))
df_mer['diff'] = np.abs(df_mer['pct_ita'] - df_mer["pct_ger"])
df_mer.sort_values('diff', ascending = False, inplace = True)
df_mer = df_mer[(~df_mer.pct_ita.isna()) | ((~df_mer.pct_ger.isna()))]
df_mer = df_mer[df_mer.age_group != '100-104']
df_mer.to_pickle("C:\\Data\\migration\\bilateral_stocks\\germany\\germany_italy_ratios.pkl")

fig, ax = plt.subplots()
plt.scatter(x = df_mer['n_people_ita'], y = df_mer["n_people_ger"])
plt.grid()
plt.show(block = True)
