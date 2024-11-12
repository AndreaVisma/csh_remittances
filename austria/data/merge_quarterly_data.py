import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import austria_nighbours, dict_names

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

## income category
df_class = pd.read_excel("c:\\data\\economic\\income_classification_countries_wb.xlsx", usecols="A:B", skipfooter=49)
df_class['country'] = df_class['country'].map(dict_names)
df = df.merge(df_class, on = 'country', how = 'left')

##GDP
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\quarterly_gdp_clean.xlsx")
for year in tqdm(df_gdp.year.unique(),
                          total = len(df_gdp.year.unique())):
    df_year = df_gdp[df_gdp.year == year]
    for quarter in df_year.quarter.unique():
        df_gdp.loc[(df_gdp.year == year) & (df_gdp.quarter == quarter), 'delta_gdp'] = (
                df_gdp.loc[(df_gdp.year == year) & (df_gdp.quarter == quarter), 'gdp_per_capita'] -
                df_gdp.loc[(df_gdp.year == year) & (df_gdp.quarter == quarter) & (df_gdp.country == 'Austria'), 'gdp_per_capita'].item())
df = df.merge(df_gdp, on = ['country', 'year', 'quarter'], how = 'left')


#natural disasters
df_nd = pd.read_excel("C:\\Data\\natural_disasters\\emdat_country_type_quarterly.xlsx")
df_nd.rename(columns = {'Country' : 'country', 'Start Year': 'year'}, inplace = True)
df_nd = df_nd[['country', 'year', 'quarter', 'total affected']].groupby(['country', 'year', 'quarter']).sum().reset_index()

df = df.merge(df_nd, on = ['country', 'year', 'quarter'], how = 'left')
df['total affected'] = df['total affected'].fillna(0)

##growth rate of remittances
df = df.sort_values(by=['country', 'year', 'quarter'])
df['growth_rate_rem'] = df.groupby('country')['remittances'].pct_change() * 100  # Multiply by 100 for percentage format
df.replace([np.inf, -np.inf], 0, inplace=True)

df.dropna(inplace =True)

df.to_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx", index = False)