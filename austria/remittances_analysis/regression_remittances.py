"""
Script: regression_remittances.py
Author: Andrea Vismara
Date: 04/10/2024
Description: try to run a regression on the panel data
"""
import numpy as np
from linearmodels.panel import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.datasets import wage_panel
import statsmodels.api as sm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel.xlsx")
df["year"] = df.year.astype(int)
df["rem_per_migrant"] = df["mln_euros"] * 1_000_000 / df["pop"]
df["rem_over_income"] = df["rem_per_migrant"] / df["income"]
df = df[df["pop"] > 0]
years = pd.Categorical(df["year"]).as_ordered()
countries = pd.Categorical(df.country)
df = df.set_index(["country", "year"])
#
# #plot crosscorrelation matrix
# corr = df.corr()
# sns.heatmap(corr, cmap = 'coolwarm',
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)
# plt.title("Correlation matrix")
# plt.show(block = True)
#
# #plot remittances v population
# sns.lmplot(df, x = "pop", y = "rem_per_migrant", hue = 'neighbour_dummy')
# plt.show(block = True)

df["year"] = years
df["country"] = countries

#plot neighbour dummy
gr = df[['pop', 'mln_euros', 'pct_cost', 'hcpi', 'gdp', 'dep_ratio',
       'neighbour_dummy', 'income', 'rem_per_migrant', 'rem_over_income']].groupby("country").mean().reset_index()
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
# Assuming `gr` is your DataFrame containing the data
fig = px.scatter(gr, x='pop', y='rem_over_income', title='Interactive Scatter Plot',hover_name='country',color='country')
fig.show()

fig, ax = plt.subplots(figsize = (9,6))
sns.regplot(df, x = "mln_euros", y = "neighbour_dummy", ax=ax)
plt.show(block = True)

df_hp = df[df["pop"] > 1000]

##regression
exog_vars = ['pop', 'pct_cost', 'gdp', 'dep_ratio', 'hcpi','neighbour_dummy', 'income']
exog = sm.add_constant(df[exog_vars])
mod = PanelOLS(df.mln_euros, exog, entity_effects=False, check_rank = False)
fe_res = mod.fit()
print(fe_res)