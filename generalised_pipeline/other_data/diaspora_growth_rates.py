

import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import dict_names
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"
from pandas.tseries.offsets import MonthEnd


## load un bilat matrix by sex
un_file = "C:\\Data\\migration\\bilateral_stocks\\un_stock_by_sex_destination_and_origin.xlsx"

df_un = pd.read_excel(un_file, sheet_name="Table 1", skiprows=10)

df_un.rename(columns={'Region, development group, country or area of origin' : 'origin',
                   'Region, development group, country or area of destination' : 'destination'}, inplace = True)

df_un['origin'] = df_un['origin'].str.replace('*', '')
df_un = df_un[df_un.origin.isin(dict_names.keys())]


df_un['destination'] = df_un['destination'].str.replace('*', '')
df_un = df_un[df_un.destination.isin(dict_names.keys())]
# df_un['destination'] = df_un.destination.map(dict_names)

df_un = df_un[["origin", "destination", 1995, 2000, 2005, 2010, 2015, 2020, 2024]]
df_un = pd.melt(df_un, id_vars=["origin", "destination"], var_name="year", value_name="n_people")
df_un.sort_values('year', inplace = True)

df = df_un.copy()
pbar = tqdm(df_un.origin.unique())
for origin in pbar:
    pbar.set_description(f"Processing {origin} ...")
    df_country = df_un[df_un.origin == origin]
    for dest in df_country.destination.unique():
        df.loc[(df.origin == origin) & (df.destination == dest), 'pct_change'] = (
            df.loc)[(df.origin == origin) & (df.destination == dest), 'n_people'].pct_change()

df = df[df["pct_change"] != np.inf].copy()
df.dropna(inplace = True)
df['yrly_growth_rate'] = df['pct_change'].apply(lambda x: (1 + x)**0.2 - 1)
df.loc[df.year == 2024, 'yrly_growth_rate'] = df.loc[df.year == 2024, 'yrly_growth_rate'].apply(lambda x: (1 + x)**0.25 - 1)

df['origin'] = df.origin.map(dict_names)
df['destination'] = df.destination.map(dict_names)
df['date'] = pd.to_datetime(df.year, format="%Y") + MonthEnd(0)

df.to_pickle("C://data//migration//stock_pct_change.pkl")

# keep only pairs with more than 100 people
df = df[df.n_people > 1000]
df = df.sort_values('origin')

## plot some results
pio.templates.default = "plotly_white"

print(f"""{round(100 * len(df[(df.yrly_growth_rate >-0.1) & (df.yrly_growth_rate < 0.1)]) / len(df), 2)}% of origin-dest pairs have yearly growth rates between -10% and +10%""")
fig = px.histogram(df, x="yrly_growth_rate")
fig.add_vline(x=0)
fig.layout.xaxis.tickformat = '.0%'
fig.update_layout(title = "Distribution of yearly growth rates, bilateral stocks of more than 1000 people")
fig.show()

fig1 = px.scatter(df, x = 'n_people', y = 'yrly_growth_rate', hover_data=['origin', 'destination', 'year'], color = 'origin', log_x=True)
fig1.add_hline(y=0.1)
fig1.add_hline(y=-0.1)
fig1.layout.yaxis.tickformat = '.0%'
fig1.update_layout(title = "Diaspora size vs. yearly growth rate of diaspora, bilateral stocks of more than 1000 people")
fig1.show()