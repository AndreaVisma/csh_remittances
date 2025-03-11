
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

countries = list(set(df_ita.origin).intersection(set(df_ger.origin)))
print("countries in ITA data and not in GER:")
print(set(df_ita.origin) - set(df_ger.origin))
print("=====================================")
print("countries in GER data and not in ITA:")
print(set(df_ger.origin) - set(df_ita.origin))

#########
for country in tqdm(countries):
    for year in df_ita.date.dt.year.unique():
        df_ger.loc[(df_ger.origin == country) & (df_ger.date.dt.year == year), "n_people"] = (100 *
            df_ger.loc[(df_ger.origin == country) & (df_ger.date.dt.year == year), "n_people"] /
            df_ger.loc[(df_ger.origin == country) & (df_ger.date.dt.year == year), "n_people"].sum())
        df_ita.loc[(df_ita.origin == country) & (df_ita.date.dt.year == year), "n_people"] = (100 *
            df_ita.loc[(df_ita.origin == country) & (df_ita.date.dt.year == year), "n_people"] /
            df_ita.loc[(df_ita.origin == country) & (df_ita.date.dt.year == year), "n_people"].sum())

df_ger = df_ger[df_ger.date.dt.year <= year]

df_mer = (df_ita[['date', 'origin', 'age_group', 'sex', 'n_people']].
          merge(df_ger[['date', 'origin', 'age_group', 'sex', 'n_people']],
                on = ['date', 'origin', 'age_group', 'sex'], how = 'inner',
                suffixes = ('_ita', '_ger')))
df_mer['diff'] = np.abs(df_mer['n_people_ita'] - df_mer["n_people_ger"])
df_mer.sort_values('diff', ascending = False, inplace = True)

fig, ax = plt.subplots()
plt.scatter(x = df_mer['n_people_ita'], y = df_mer["n_people_ger"])
plt.grid()
plt.show(block = True)
