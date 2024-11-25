import pandas as pd
from utils import dict_names

df = pd.read_excel("c:\\data\\population\\population_by_country_wb_raw.xls",
                   skiprows = 3)
df.set_index('Country Name', inplace=True)
df = df.iloc[:, 3:].reset_index().rename(columns={df.index.name:'country'})

df = pd.melt(df, id_vars='country', value_vars=df.columns[1:], var_name='year', value_name='population')
print(set(df.country) - set(dict_names.keys()))
df['country'] = df['country'].map(dict_names)
df.dropna(inplace = True)

df.to_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx", index = False)