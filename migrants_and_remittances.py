"""
Script: migrants_and_remittances.py
Author: Andrea Vismara
Date: 04/07/2024
Description:
"""

#import
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import geopandas as gpd
import matplotlib as mpl
mpl.use("Qtagg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.renderers.default = "browser"

### globals and utils
from utils import *

###########
# load migrants stocks data
##########
data_folder = os.getcwd() + "\\data_downloads\\data\\"

#all data
df_all = pd.read_excel(data_folder + "undesa_pd_2020_ims_stock_by_sex_destination_and_origin.xlsx",
                       skiprows=10, usecols="B,F,H:N", sheet_name="Table 1")
df_all.rename(columns={'Region, development group, country or area of destination': 'destination_mig',
                   'Region, development group, country or area of origin': 'origin_mig'}, inplace=True)

df_all['origin_mig'] = df_all['origin_mig'].apply(lambda x: remove_asterisc(x))
df_all['origin_mig'] = clean_country_series(df_all['origin_mig'])
df_all['destination_mig'] = df_all['destination_mig'].apply(lambda x: remove_asterisc(x))
df_all['destination_mig'] = clean_country_series(df_all['destination_mig'])
df_all.dropna(inplace = True)
df_all = df_all[["origin_mig", "destination_mig", 2020]].copy()
df_all.rename(columns = {2020: "migrants_2020"}, inplace = True)

###########
# load remittances flows data
##########
df_rem = pd.read_excel(data_folder + "\\bilateral_remittance_matrix_2021.xlsx",
                   skiprows = 1, nrows = 215)
df_rem.rename(columns = {df_rem.columns[0]: "origin_rem"}, inplace = True)
df_rem = pd.melt(df_rem, id_vars=['origin_rem'], value_vars=df_rem.columns.tolist()[1:])
df_rem.dropna(inplace = True)
df_rem.rename(columns = {"variable" : "destination_rem", "value" : "mln_remittances"}, inplace = True)
df_rem['origin_rem'] = clean_country_series(df_rem['origin_rem'])
df_rem['destination_rem'] = clean_country_series(df_rem['destination_rem'])
df_rem.dropna(inplace = True)

##merge
df = df_all.merge(df_rem, left_on = ["origin_mig", "destination_mig"], right_on = ["destination_rem", "origin_rem"])
df["remit_per_migrant"] = 1e6 * df.mln_remittances / df.migrants_2020
df = df[df["remit_per_migrant"] < np.inf]

###plot some charts
#logs
#by origin
df = df.sort_values("origin_mig")
fig = px.scatter(df, x="migrants_2020", y="mln_remittances", color="origin_mig",
                 hover_data=['remit_per_migrant', "destination_mig"],
                 title= "remittances per migrant by country of origin of migrants (which is the destination of remittances)",
                 log_x=True, log_y=True)
fig.update_layout(yaxis_title = "Remittances in 2021 (USD)", xaxis_title = "Migrant stock in 2020 (nr people)")
fig.show()

df = df.sort_values("origin_mig")
fig = px.scatter(df, y="remit_per_migrant", color="origin_mig",
                 hover_data=['remit_per_migrant', "destination_mig", "origin_rem", "destination_rem"],
                 title= "remittances per migrant by country of origin of migrants (which is the destination of remittances)")
fig.update_layout(yaxis_title = "Remittances per migrant in 2021 (USD)")
fig.show()

# #by destination
# df = df.sort_values("destination")
# fig = px.scatter(df, x="migrants_2020", y="mln_remittances", color="destination",
#                  hover_data=['remit_per_migrant', "origin"],
#                  title= "remittances per migrant by destination",
#                  log_x=True, log_y=True)
# fig.show()

## hist
fig = px.histogram(df[df.remit_per_migrant < 150_000], x="remit_per_migrant")
fig.show()

ax = df[df.remit_per_migrant < 100_000]["remit_per_migrant"].plot.kde()
plt.show(block = True)

## high remittances
fig = px.scatter(df[df.remit_per_migrant > 10_000], x = "migrants_2020", y ="remit_per_migrant",
                 color = "origin_mig", log_x=True, log_y=True,
                 hover_data=['remit_per_migrant', "destination_mig", "origin_rem", "destination_rem"],
                 title= "remittances per migrant by origin")
fig.show()



