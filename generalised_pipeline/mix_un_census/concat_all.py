
import pandas as pd
import numpy as np
import re

latam = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\latam\\processed_latam_hosts_3obs.pkl")
euro = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\europe\\processed_european_hosts_3obs.pkl")
row = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\world\\processed_row_hosts_3obs.pkl")
##
df_us = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\us\\processed_usa.pkl")
df_us = df_us[df_us.date.isin(["2010-01-31", "2015-01-31", "2020-01-31"])]
df_us['mean_age'] = df_us['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
df_us["age_group"] = pd.cut(df_us.mean_age, bins = bins).map(dict(zip(bins, labels)))
df_us = df_us[['date', 'origin', 'destination', 'age_group', 'sex', 'n_people']]

#
ger = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\germany\\processed_germany.pkl")
ger = ger[ger.date.isin(["2010-01-31", "2015-01-31", "2020-01-31"])]
ger['mean_age'] = ger['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
ger["age_group"] = pd.cut(ger.mean_age, bins = bins).map(dict(zip(bins, labels)))
ger = ger[['date', 'origin', 'destination', 'age_group', 'sex', 'n_people']]

ita = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\italy\\processed_italy.pkl")
ita = ita[ita.date.isin(["2010-01-31", "2015-01-31", "2020-01-31"])]
ita['mean_age'] = ita['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
ita["age_group"] = pd.cut(ita.mean_age, bins = bins).map(dict(zip(bins, labels)))
ita = ita[['date', 'origin', 'destination', 'age_group', 'sex', 'n_people']]

##
df_all = pd.concat([latam, euro, row, df_us, ger, ita])
df_all.to_pickle("C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_3obs.pkl")
