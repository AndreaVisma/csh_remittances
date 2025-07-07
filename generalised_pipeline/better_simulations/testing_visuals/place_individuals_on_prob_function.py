
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import random
import time
import itertools
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from random import sample
from pandas.tseries.offsets import MonthEnd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)
from utils import zero_values_before_first_positive_and_after_first_negative


## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors.pkl"
df = pd.read_pickle(diasporas_file)
df = df[(df.date.dt.year == 2018) & (df.date.dt.month == 1)]
df = df.dropna()

##nta accounts
df_nta = pd.read_pickle("C:\\Data\\economic\\nta\\processed_nta.pkl")
nta_dict = {}

###gdp to infer remittances amount
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx").groupby('country').mean().reset_index().rename(columns = {'country' : 'origin'}).drop(columns = 'year')

## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags.pkl")

####
df_f = df[df.age_group == '30-34']

results_dfs = []
for country in tqdm(df_f.destination.unique()):
    df_country = df_f[df_f.destination == country].copy()
    df_nta_ita = df_nta.query(f"""`country` == '{country}'""")[['age', 'nta']].fillna(0)
    for ind, row in df_nta_ita.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)
    df_country['theta'] = (param_nta * (nta_dict[30] - 1))  \
                + (param_asy * df_country['asymmetry']) + (param_gdp * df_country['gdp_diff_norm'])
    df_country['probability'] = 1 / (1 + np.exp(-df_country['theta']))
    results_dfs.append(df_country)

df_res = pd.concat(results_dfs)
df_res.sort_values('n_people', inplace = True, ascending = False)

###### plot
import seaborn as sns
sns.set_theme(style="white")
max_val_size = 0.005 * (df_res.n_people.max() / df_res.n_people.min())
wanted_couples = [('Mexico', 'USA'), ('India', 'United Arab Emirates'), ('Yemen', 'Saudi Arabia'), ('Bangladesh', 'India'),
                  ('Syria', 'Turkey'), ('Monaco', 'Germany')]

fig, ax = plt.subplots(figsize = (7.5,9))
df_plot_list = []
for i in range(len(wanted_couples)):
    df_plot = df_res[(df_res.origin == wanted_couples[i][0]) & (df_res.destination == wanted_couples[i][1])]
    df_plot_list.append(df_plot)
df_plot = pd.concat(df_plot_list)

sns.scatterplot(x="theta", y="probability", size="n_people",
            sizes=(20, max_val_size), alpha=.4, hue = 'origin',
                data=df_res, ax = ax, palette=["grey"], legend=False)
sns.scatterplot(x="theta", y="probability", size="n_people",
            sizes=(20, max_val_size), alpha=0.8, hue = 'origin',
                data=df_plot, ax = ax, legend=False)
plt.grid(True)
fig.savefig('.\plots\\for_paper\\individuals_probability_30yrsold.png', bbox_inches = 'tight')
plt.show(block = True)

################ average