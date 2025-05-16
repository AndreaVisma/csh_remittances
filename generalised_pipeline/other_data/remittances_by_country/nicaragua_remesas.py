

import pandas as pd
import numpy as np

file = "C:\\Data\\remittances\\Nicaragua\\remesas_nicaragua.xlsx"
df = pd.read_excel(file, skiprows=4, usecols="B:I", skipfooter=15)
dict_meses = dict(zip(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep',
       'Oct', 'Nov', 'Dic'], [x for x in range(1, 13)]))

#clean dates
df['mes'] = df['year'].str[:3].map(dict_meses)
df['year'] = "20" + df['year'].str[-2:]
df.dropna(inplace = True)
df['year'] = df['year'].astype(int)
df['date'] = pd.to_datetime({
    'year': df['year'],
    'month': df['mes'],
    'day': 1
}) + pd.offsets.MonthEnd(0)

# melt
countries = ['USA', 'Spain', 'Costa Rica', 'Panama', 'Canada', 'Mexico', 'El Salvador']
df = pd.melt(df, id_vars='date', value_vars=countries, var_name='destination', value_name='remittances')
df['origin'] = "Nicaragua"
df['remittances'] *= 1_000_000

df.to_pickle("C:\\Data\\remittances\\Nicaragua\\nic_remittances_detail.pkl")