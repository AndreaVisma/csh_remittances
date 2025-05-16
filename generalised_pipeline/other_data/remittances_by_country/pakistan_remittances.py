
import pandas as pd
import numpy as np
from utils import dict_names

file = "C:\\Data\\remittances\\Pakistan\\pakistan_remittances.csv"
df = pd.read_csv(file)[['Observation Date', 'Series Display Name', 'Observation Value', 'Unit']]
df.rename(columns = dict(zip(['Observation Date', 'Series Display Name', 'Observation Value', 'Unit'],
                             ['date', 'destination', 'remittances', 'unit'])), inplace = True)
df['origin'] = "Pakistan"
## clean destination
df['destination'] = df['destination'].str.replace(".", "")
df['destination'] = df['destination'].str.replace("\d+", "", regex=True)
df['destination'] = df['destination'].str.strip()
print(set(df.destination) - set(dict_names.keys()))
df['destination'] = df.destination.map(dict_names)
df.dropna(inplace = True)

#clean the rest
df['date'] = pd.to_datetime(df['date'])
df['remittances'] *= 1_000_000
df.drop(columns = 'unit', inplace = True)

##save
df.to_pickle("C:\\Data\\remittances\\Pakistan\\pak_remittances_detail.pkl")