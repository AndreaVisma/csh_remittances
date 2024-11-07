import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import os
import matplotlib.ticker as mtick
import geopandas
from utils import dict_names
from tqdm import tqdm
import country_converter as coco
cc = coco.CountryConverter()
import plotly.io as pio
pio.renderers.default = 'browser'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_excel("c:\\data\\remittances\\austria\\quarterly_remittances.xlsx", skiprows=3, skipfooter=3, sheet_name="Tabelle2")
df.columns = ['y-q', 'country', 'remittances']
df['year'] = df['y-q'].apply(lambda x: int(x[:4]))
df['quarter'] = df['y-q'].apply(lambda x: int(x[-1:]))

df['country'].fillna('NA', inplace= True)
df.country = df.country.apply(lambda x: cc.convert(names=[x], to = 'name'))
df.country = df.country.map(dict_names)

df = df.sort_values(['country', 'year', 'quarter'])
df.remittances = df.remittances.ffill()
df.dropna(inplace = True)
df['date'] =  pd.to_datetime(df['y-q'])
df.drop(columns='y-q', inplace = True)

df.to_excel("c:\\data\\remittances\\austria\\quarterly_remittances_sent_clean.xlsx")

def plot_remittances_country(country):

    df_country = df[df.country == country]

    fig, ax = plt.subplots(figsize = (9,6))
    ax.plot(df_country.date, df_country.remittances)
    plt.xlabel('Quarters')
    plt.ylabel('Remittances in EUR')
    plt.grid()
    plt.title(f'Remittances sent to {country}')
    plt.show(block = True)

plot_remittances_country('Czechia')