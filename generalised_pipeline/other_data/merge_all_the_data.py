

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time
import itertools
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from random import sample
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_interpolated.pkl"
df = pd.read_pickle(diasporas_file)
df = df[df.n_people > 0]

##exponential betas for years of stay
# df_betas = pd.read_pickle("C:\\Data\\migration\\simulations\\exponential_betas.pkl")

## family asymmetry
asymmetry_file = "C:\\Data\\migration\\bilateral_stocks\\pyramid_asymmetry_beginning_of_the_year.pkl"
asy_df = pd.read_pickle(asymmetry_file)

## diaspora growth rates
# growth_rates = pd.read_pickle("C://data//migration//stock_pct_change.pkl")

## gdp differential
df_gdp = (pd.read_pickle("c:\\data\\economic\\gdp\\annual_gdp_deltas.pkl"))
df_gdp['gdp_diff_norm'] = 2* (df_gdp['gdp_diff'] - df_gdp['gdp_diff'].min()) / (df_gdp['gdp_diff'].max() - df_gdp['gdp_diff'].min()) - 1

##nta accounts
df_nta = pd.read_pickle("C:\\Data\\economic\\nta\\processed_nta.pkl")
nta_dict = {}

## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags.pkl")

df = df[df.destination != "Cote d'Ivoire"]
list_dfs = []
for country in tqdm(df.destination.unique()):
    countries_or = (df[df.destination == country]['origin'].unique().tolist())
    df_country_ita = df.query(f"""`origin` in {countries_or} and `destination` == '{country}'""")
    df_country_ita = df_country_ita[['date', 'origin', 'age_group', 'mean_age', 'destination', 'n_people']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).sum().reset_index()
    # asy
    asy_df_ita = asy_df.query(f"""`destination` == '{country}'""")
    df_country_ita = df_country_ita.sort_values(['origin', 'date']).sort_values(['origin', 'date']).merge(asy_df_ita[["date", "asymmetry", "origin"]],
                                  on=["date", "origin"], how='left').ffill()

    ##gdp diff
    df_gdp_ita = df_gdp.query(f"""`destination` == '{country}'""")
    df_country_ita = df_country_ita.merge(df_gdp_ita[["date", "gdp_diff_norm", "origin"]], on=["date", "origin"],
                                  how='left')
    df_country_ita['gdp_diff_norm'] = df_country_ita['gdp_diff_norm'].bfill()

    list_dfs.append(df_country_ita)
df_all = pd.concat(list_dfs)
df_all.to_pickle("C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors.pkl")
