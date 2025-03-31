
import pandas as pd
import numpy as np
import re
from utils import dict_names

nta_file = "C:\\Data\\economic\\nta\\NTA profiles.xlsx"

c_vals = pd.read_excel(nta_file, sheet_name='C')
yl_vals = pd.read_excel(nta_file, sheet_name='YL')

def work_df(df):
    cols = df.columns[11:].tolist()
    # cols = [re.findall(r'\d+', x) for x in cols]
    cols.append('country')
    df = df[cols]
    df = pd.melt(df, id_vars='country', value_name='value', var_name='age')
    df['age'] = df['age'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
    df['country'] = df.country.map(dict_names)
    return df

c_vals = work_df(c_vals).rename(columns = {'value' : 'consumption'})
yl_vals = work_df(yl_vals).rename(columns = {'value' : 'labour_income'})

df = c_vals.merge(yl_vals, on = ['country', 'age'])
df['nta'] = df['labour_income'] / df['consumption']

wb_groups = pd.read_excel("C://data//economic//income_classification_countries_wb.xlsx")[['country', 'group']]
wb_groups['country'] = wb_groups.country.map(dict_names)
df = df.merge(wb_groups, on = 'country', how = 'left')
print(df.isna().sum())
df.to_csv("C:\\Data\\economic\\nta\\processed_nta.csv")
