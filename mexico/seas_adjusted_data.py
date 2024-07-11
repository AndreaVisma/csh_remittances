"""
Script: mexico_remittances_explore.py
Author: Andrea Vismara
Date: 10/07/2024
Description: Explores the data for the remittances inflow in mexico
"""

##imports
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.io as pio
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
fig.to_html(out_folder + "total_adjusted.html")
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
fig.to_html(out_folder + "operations_adjusted.html")
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
fig.to_html(out_folder + "promedio_adjusted.html")
fig.show()