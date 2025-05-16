
import pandas as pd
import numpy as np

file = "C:\\Data\\remittances\\Guatemala\\remfam2010_2021.xls"
df = pd.read_excel(file, skiprows=9, skipfooter=23, usecols="B:V")

dict_meses = dict(zip(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio',
       'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'],
                      [x for x in range(1,13)]))
df['Mes'] = df.Mes.map(dict_meses)

df = pd.melt(df, id_vars='Mes', value_vars=df.columns[1:], var_name='year', value_name='remittances')
df['date'] = pd.to_datetime({
    'year': df['year'],
    'month': df['Mes'],
    'day': 1
}) + pd.offsets.MonthEnd(0)

df['origin'] = "Guatemala"
df['destination'] = "USA"

df=df[['date', 'origin', 'destination', 'remittances']].copy()
df['remittances'] *= (1_000_000 * 0.97)
df.dropna(inplace = True)

##save
df.to_pickle("C:\\Data\\remittances\\Guatemala\\gua_remittances_detail.pkl")