"""
Script: correct_exchange_rate.py
Author: Andrea Vismara
Date: 29/07/2024
Description: Explores the data for the remittances inflow in mexico, correcting for the exchange rate
"""

import pandas as pd
import datetime
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.io as pio
from utils import *
pio.renderers.default = "browser"

out_folder = "c:\\git-projects\\csh_remittances\\mexico\\plots\\"

df = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_seasonally_adjusted.xlsx")

ex_rate = pd.read_excel("c:\\data\\economic\\mexico\\peso_usd_exrate.xls",
                        skiprows = 10)
ex_rate = ex_rate.groupby([ex_rate.observation_date.dt.month, ex_rate.observation_date.dt.year]).head(1).reset_index(drop = True)
ex_rate["observation_date"] = ex_rate["observation_date"].apply(lambda x: datetime.datetime(x.year, x.month, 1, 0, 0))
ex_rate.rename(columns={"observation_date":"date", "DEXMXUS" : "pesos_for_dollar"}, inplace = True)
ex_rate.loc[ex_rate.pesos_for_dollar == 0, "pesos_for_dollar"] = np.nan
ex_rate["pesos_for_dollar"] = ex_rate["pesos_for_dollar"].interpolate()
ex_rate.isna().sum()

df = df.merge(ex_rate, on = "date", how= "left")

#### plot both series
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df.date, y=df.total_mln_seas, name = "total remittances to Mexico"),
              secondary_y=False)
fig.add_trace(go.Scatter(x=df.date, y=df.pesos_for_dollar, name = "Pesos for 1 USD"),
              secondary_y=True)
fig['layout']['yaxis2']['showgrid'] = False
fig.update_layout(title = "Total remittances to Mexico and dollar-peso exchange rate")
fig.write_html(out_folder + "total_remittances_and_exchange_rate.html")
fig.show()

#plot total in pesos and in dollars
df["total_remittances_USD"] = df.total_mln_seas / df.pesos_for_dollar

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df.date, y=df.total_mln_seas, name = "total remittances to Mexico<br><b>in Mexican pesos<b>"),
              secondary_y=False)
fig.add_trace(go.Scatter(x=df.date, y=df.total_remittances_USD, name = "total remittances to Mexico<br><b>in US dollars<b>"),
              secondary_y=True)
fig['layout']['yaxis2']['showgrid'] = False
fig.update_layout(title = "Total remittances to Mexico in pesos and dollars")
fig.update_yaxes(title = "mln Mexican pesos", secondary_y=False)
fig.update_yaxes(title = "mln US dollars", secondary_y=True)
fig.write_html(out_folder + "total_remittances_in_pesos_dollars.html")
fig.show()