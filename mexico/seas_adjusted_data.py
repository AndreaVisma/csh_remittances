"""
Script: mexico_remittances_explore.py
Author: Andrea Vismara
Date: 10/07/2024
Description: Explores the data for the remittances inflow in mexico
"""

##imports
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.io as pio
from utils import *
pio.renderers.default = "browser"

out_folder = "c:\\git-projects\\csh_remittances\\mexico\\plots\\seasonal_adjustment\\"

df = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_seasonally_adjusted.xlsx")

## total amounts
fig = go.Figure()
fig.add_trace(go.Scatter(
    x = df.date, y = df.total_mln, name="total remittances<br>original series"
))
fig.add_trace(go.Scatter(
    x = df.date, y = df.total_mln_seas, name="total remittances<br>seasonally adjusted"
))
fig.update_layout(title = "Seasonally adjusted remittances amounts")
fig.write_html(out_folder + "total_adjusted.html")
fig.show()

## total operations
fig = go.Figure()
fig.add_trace(go.Scatter(
    x = df.date, y = df.total_operations, name="total operations<br>original series"
))
fig.add_trace(go.Scatter(
    x = df.date, y = df.total_operations_seas, name="total operations<br>seasonally adjusted"
))
fig.update_layout(title = "Seasonally adjusted number of operations")
fig.write_html(out_folder + "operations_adjusted.html")
fig.show()

## promedio
fig = go.Figure()
fig.add_trace(go.Scatter(
    x = df.date, y = df.total_mean_op, name="mean per operation<br>original series"
))
fig.add_trace(go.Scatter(
    x = df.date, y = df.total_mean_op_seas, name="mean per operation<br>seasonally adjusted"
))
fig.update_layout(title = "Seasonally adjusted mean dollars per operation")
fig.write_html(out_folder + "promedio_adjusted.html")
fig.show()

# operations and total
## total amounts
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(
    x = df.date, y = df.total_mln_seas, name="total remittances<br>seasonally adjusted (lhs)"
), secondary_y=False)
# fig.update_layout(title = "Seasonally adjusted remittances amounts")
## total operations
fig.add_trace(go.Scatter(
    x = df.date, y = df.total_operations_seas, name="total operations<br>seasonally adjusted (rhs)"
), secondary_y=True)
fig.add_trace(go.Scatter(
    x = df.date, y = df.total_mean_op_seas * 20, name=" 20 x mean per operation<br>seasonally adjusted (rhs)"
), secondary_y=True)
fig.update_layout(title = "Seasonally adjusted total remittances and number of operations")
fig.update_yaxes(title_text="Millions USD in remittances sent to Mexico", secondary_y=False)
fig.update_yaxes(title_text="Thousands of remittances operations", secondary_y=True)
fig['layout']['yaxis2']['showgrid'] = False
fig.write_html(out_folder + "total_and_operations_adjusted.html")
fig.show()

##################
# add migration stock data
#################
data_folder = os.getcwd() + "\\data_downloads\\data\\"

#all data
df_all = pd.read_excel(data_folder + "undesa_pd_2020_ims_stock_by_sex_destination_and_origin.xlsx",
                       skiprows=10, usecols="B,F,H:N", sheet_name="Table 1")
df_all.rename(columns={'Region, development group, country or area of destination': 'destination',
                   'Region, development group, country or area of origin': 'origin'}, inplace=True)
df_all['origin'] = df_all['origin'].apply(lambda x: remove_asterisc(x))
df_all['origin'] = clean_country_series(df_all['origin'])
df_all['destination'] = df_all['destination'].apply(lambda x: remove_asterisc(x))
df_all['destination'] = clean_country_series(df_all['destination'])
df_all.dropna(inplace = True)

df_mex = df_all[(df_all.origin == "Mexico") & (df_all.destination == "USA")]
df_mex = pd.melt(df_mex, id_vars=['origin', 'destination'], value_vars=[x for x in df_mex.columns[2:]],
                 var_name='year', value_name='migrants_stock')
df_mex = df_mex[df_mex.year > 1990]
df_mex.loc[len(df_mex.index)] = ['Mexico', 'USA', 2022, 10_820_514]

df_usa = df_all[(df_all.origin == "USA") & (df_all.destination == "Mexico")]
df_usa = pd.melt(df_usa, id_vars=['origin', 'destination'], value_vars=[x for x in df_usa.columns[2:]],
                 var_name='year', value_name='migrants_stock')
df_usa = df_usa[df_usa.year > 1990]

df = df[df.date < "2022"]

## total amounts
fig = make_subplots(specs=[[{"secondary_y": True}]])
##remittances
fig.add_trace(go.Scatter(
    x = df.date, y = df.total_mln, name="total remittances<br>original series (lhs)"
), secondary_y= False)
fig.add_trace(go.Scatter(
    x = df.date, y = df.total_mln_seas, name="total remittances<br>seasonally adjusted (lhs)"
), secondary_y= False)
##migrants
fig.add_trace(go.Scatter(
    x = df_mex.year, y = df_mex.migrants_stock, name="Mexican-born migrants<br>in the United States (rhs)",
    mode = 'lines'
), secondary_y= True)
fig.update_layout(title = "Remittances to Mexico and Mexican migrants in the United States")
fig.update_yaxes(title_text="Millions USD in remittances sent to Mexico", secondary_y=False)
fig.update_yaxes(title_text="Number of mexican-born migrants in the US", secondary_y=True)
fig['layout']['yaxis2']['showgrid'] = False
fig.update_yaxes(range=[6_500_000, 14_000_000], secondary_y = True)
fig.write_html(out_folder + "remittances_and_migrants.html")
fig.show()

