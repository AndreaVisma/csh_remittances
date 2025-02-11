"""
Script: plot_rem_hist.py
Author: Andrea Vismara
Date: 10/02/2025
Description:
"""

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
mpl.use("Qtagg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

# define global variables
data_folder = os.getcwd() + "\\data_downloads\\data\\"

######
# Load all the data in
######

#load inflow of remittances
df_in = pd.read_excel(data_folder + "inward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y")
df_in.rename(columns = {"Remittance inflows (US$ million)": "country"}, inplace=True)
df_in = pd.melt(df_in, id_vars=['country'], value_vars=df_in.columns.tolist()[1:])
df_in.rename(columns = {"variable": "year", "value" : "inflow"}, inplace=True)
df_in.year = df_in.year.astype('int')

#load outflow of remittances
df_out = pd.read_excel(data_folder + "outward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y", skiprows=2)
df_out.rename(columns = {"Remittance outflows (US$ million)": "country"}, inplace=True)
df_out = pd.melt(df_out, id_vars=['country'], value_vars=df_out.columns.tolist()[1:])
df_out.rename(columns = {"variable": "year", "value" : "outflow"}, inplace=True)
df_out.replace({"2023e": '2023'}, inplace =True)
df_out.year = df_out.year.astype('int')

df = df_in.merge(df_out, on = ['country', 'year'], how = 'left')
df.dropna(inplace = True)
df['inflow'] /= 1_000
df['outflow'] /= 1_000

df_2023 = df[df['year'] == 2023]
top_receivers = df_2023.nlargest(5, 'inflow')
top_senders = df_2023.nlargest(5, 'outflow')

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,6), sharex=True)
ax1.barh(top_senders['country'], top_senders['outflow'], color='red')
ax2.barh(top_receivers['country'], top_receivers['inflow'], color='blue')
fig.savefig("C:\\git-projects\\csh_remittances\\plots\\barchart_flows.svg", bbox_inches = 'tight')
plt.show(block = True)


########
