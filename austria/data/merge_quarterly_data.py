import pandas as pd
import numpy as np
from utils import austria_nighbours

#population
df_pop = pd.read_excel("c:\\data\\migration\\austria\\quarterly_population_clean.xlsx")

#remittances
df_rem = pd.read_excel("c:\\data\\remittances\\austria\\quarterly_remittances_sent_clean.xlsx")
df = df_pop.merge(df_rem, on = ['country', 'year', 'quarter'], how = 'inner')

#dependency_ratio
df_age = pd.read_excel("c:\\data\\population\\austria\\age_nationality_hist_quarterly.xlsx")
df = df.merge(df_age, on = ['country', 'year', 'quarter'], how = 'inner')

# dummy for neighbouring countries
df["neighbour_dummy"] = np.where(df["country"].isin(austria_nighbours), 1, 0)

#natural disasters
df_nd = pd.read_excel("C:\\Data\\natural_disasters\\emdat_country_type_quarterly.xlsx")
df_nd.rename(columns = {'Country' : 'country', 'Start Year': 'year'}, inplace = True)
df_nd = df_nd[['country', 'year', 'quarter', 'total affected']].groupby(['country', 'year', 'quarter']).sum().reset_index()

df = df.merge(df_nd, on = ['country', 'year', 'quarter'], how = 'left')
df['total affected'] = df['total affected'].fillna(0)

df.dropna(inplace =True)

df.to_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx", index = False)