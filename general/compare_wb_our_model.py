
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
import seaborn as sns
from random import sample, uniform
import random
from pandas.tseries.offsets import MonthEnd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)
from utils import zero_values_before_first_positive_and_after_first_negative

#### gdp and remittances
df_gdp_or = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx").rename(columns={'country' : 'origin', 'gdp' : 'gdp_or'})
df_gdp_dest = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx").rename(columns={'country' : 'destination', 'gdp' : 'gdp_dest'})

df_rem_ita = pd.read_parquet("C:\\Data\\my_datasets\\italy\\simulation_data.parquet")
df_rem_ita['destination'] = 'Italy'
df_rem_ita.rename(columns = {"country": 'origin'}, inplace = True)
df_rem_ita = df_rem_ita[~df_rem_ita[["date", "origin"]].duplicated()][
    ["date", "origin", "destination", "remittances"]]
# PHIL
df_rem_phil = pd.read_pickle("C:\\Data\\remittances\\Philippines\\phil_remittances_detail.pkl")
# PAK
df_rem_pak = pd.read_pickle("C:\\Data\\remittances\\Pakistan\\pak_remittances_detail.pkl")
# GUA
df_rem_gua = pd.read_pickle("C:\\Data\\remittances\\Guatemala\\gua_remittances_detail.pkl")
# GUA
df_rem_nic = pd.read_pickle("C:\\Data\\remittances\\Nicaragua\\nic_remittances_detail.pkl")
# MEX
df_rem_mex = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_renamed.xlsx")[["date", "total_mln"]]
df_rem_mex['date'] = pd.to_datetime(df_rem_mex['date'], format="%Y%m") + MonthEnd(0)
df_rem_mex['origin'] = "Mexico"
df_rem_mex['destination'] = "USA"
df_rem_mex.rename(columns = {'total_mln' : 'remittances'}, inplace = True)
df_rem_mex['remittances'] *= 1_000_000

df_rem = pd.concat([df_rem_ita, df_rem_phil, df_rem_mex, df_rem_pak, df_rem_gua, df_rem_nic])
df_rem['date'] = pd.to_datetime(df_rem.date)
df_rem['year'] = df_rem.date.dt.year

df_wb = df_rem.merge(df_gdp_or, on = ['origin', 'year'], how = 'left')
df_wb = df_wb.merge(df_gdp_dest, on = ['destination', 'year'], how = 'left')

beta = 0
df_wb['r_factor'] = 0
df_wb.loc[df_wb.gdp_dest < df_wb.gdp_or, 'r_factor'] = df_wb.loc[df_wb.gdp_dest < df_wb.gdp_or, 'gdp_or'] / 12
df_wb.loc[df_wb.gdp_dest >= df_wb.gdp_or, 'r_factor'] = (df_wb.loc[df_wb.gdp_dest >= df_wb.gdp_or, 'gdp_or'] +
    (df_wb.loc[df_wb.gdp_dest >= df_wb.gdp_or, 'gdp_dest'] - df_wb.loc[df_wb.gdp_dest >= df_wb.gdp_or, 'gdp_or'])**beta) / 12

#########
# migrants
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors.pkl"
df = pd.read_pickle(diasporas_file)
df = df[["date", "origin", "destination", "n_people"]].groupby(["date", "origin", "destination"]).sum().reset_index()

df_wb['date'] = pd.to_datetime(df_wb.date)
df_wb = df_wb.merge(df, on = ["date", "origin", "destination"], how = 'left')
df_wb.dropna(inplace = True)

df_wb['sim_remittances'] = df_wb['n_people'] * df_wb['r_factor']
df_wb = df_wb[df_wb.origin != "Libya"]
df_wb_yearly = df_wb[['year', 'origin', 'destination', 'remittances', 'sim_remittances']].groupby(['year', 'origin', 'destination']).mean().reset_index()

################################

df_results = pd.read_parquet("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_with_disasters_CORRECT.parquet")
df_results = df_results.merge(df_rem, on =["origin", "destination", "date"], how = 'inner')
df_results = df_results.merge(df_wb[["date", "origin", "destination", "sim_remittances"]],
                              on = ["date", "origin", "destination"], how = 'inner',
                              suffixes = ("_our_model", "_wb"))
df_res_group = (df_results[['origin', 'destination', 'sim_remittances_our_model',
                           'sim_remittances_wb', 'remittances']].groupby(['origin', 'destination'])
                .sum().reset_index())
cols = ['sim_remittances_our_model', 'sim_remittances_wb', 'remittances']
for col in cols:
    df_res_group[col] /= 1e9

df_res_group['error_our_model'] = ((df_res_group['remittances'] - df_res_group['sim_remittances_our_model']))**2
df_res_group['error_wb'] = ((df_res_group['remittances'] - df_res_group['sim_remittances_wb']))**2
df_res_group['rel_err_our'] = 100 * abs((df_res_group['remittances'] - df_res_group['sim_remittances_our_model'])/
                               df_res_group['remittances'])
df_res_group['rel_err_wb'] = 100 * abs((df_res_group['remittances'] - df_res_group['sim_remittances_wb'])/
                               df_res_group['remittances'])
df_plot = df_res_group[(df_res_group.error_our_model < 200) & (df_res_group.error_wb < 200)]
#######################
list_res = []
for i in tqdm(range(1000)):
    df_sam = df_plot.sample(int(0.6 * len(df_res_group)))
    our_err = df_sam.error_our_model.mean()
    wb_err  = df_sam.error_wb.mean()
    our_err_rel = df_sam.rel_err_our.mean()
    wb_err_rel  = df_sam.rel_err_wb.mean()
    local_errors = [i, our_err, wb_err, our_err_rel, wb_err_rel]
    list_res.append(local_errors)

df_sam_tot = pd.DataFrame(list_res, columns=["run_nr", "error_our_model", "error_wb", "rel_our", "rel_wb"])

# histogram errors
plt.figure(figsize=(10, 6))
plt.hist(df_sam_tot['error_our_model'], bins = 20,alpha=0.7, label='Our Error')
plt.hist(df_sam_tot['error_wb'], bins = 20, alpha=0.7, label='WB Error')
plt.xlabel('Error Value')
plt.ylabel('Frequency')
plt.title('Histogram of Errors')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block = True)

import seaborn as sns

df_melted = df_sam_tot.melt(id_vars=['run_nr'], value_vars=['error_our_model', 'error_wb'],
                    var_name='Error Type', value_name='Error Value')

# Plot with seaborn

fig, ax = plt.subplots(figsize=(10, 6))

# Histogram only (no kde here)
sns.histplot(
    data=df_melted,
    x='Error Value',
    hue='Error Type',
    bins=15,
    element='step',
    stat='density',
    common_norm=False,
    ax=ax
)

# Add KDEs separately with thicker lines
sns.kdeplot(
    data=df_melted,
    x="Error Value",
    hue="Error Type",
    common_norm=False,
    ax=ax,
    lw=2.5  # works here
)

# Add vertical lines at the means for each Error Type
palette = sns.color_palette()  # get seaborn colors
for i, (error_type, group_data) in enumerate(df_melted.groupby("Error Type")):
    mean_val = group_data["Error Value"].mean()
    ax.axvline(mean_val, linestyle="--", linewidth=2, color=palette[i],
               label=f"{error_type} Mean")

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.grid(True)
plt.tight_layout()
plt.legend().remove()
# ax.legend(False)
fig.savefig('./plots/for_paper/comparison_WB.svg', bbox_inches='tight')
plt.show(block=True)

df_melted_2 = df_plot.melt(id_vars=['origin', 'destination'], value_vars=['error_our_model', 'error_wb'],
                    var_name='Error Type', value_name='Error Value')
plt.figure(figsize=(10, 6))
sns.histplot(data=df_melted_2, x='Error Value', hue='Error Type', bins=15, kde=True, element='step', stat='density', common_norm=False)
plt.title('Histogram of Errors (Seaborn)')
plt.grid(True)
plt.tight_layout()
plt.show(block = True)













######################
list_res = []
for i in tqdm(range(1000)):
    df_sam = df_res_group.sample(int(0.4 * len(df_res_group)))
    our_err = df_sam.error_our_model.sum()
    wb_err  = df_sam.error_wb.sum()
    our_err_rel = df_sam.rel_err_our.sum()
    wb_err_rel  = df_sam.rel_err_wb.sum()
    local_errors = [i, our_err, wb_err, our_err_rel, wb_err_rel]
    list_res.append(local_errors)

df_sam_tot = pd.DataFrame(list_res, columns=["run_nr", "our_err", "wb_err", "rel_our", "rel_wb"])

plt.figure(figsize=(10, 6))
plt.hist(df_plot['rel_err_our'], bins = 50,alpha=0.7, label='Our Error')
plt.hist(df_plot['rel_err_wb'], bins = 50, alpha=0.7, label='WB Error')
plt.xlabel('Error Value')
plt.ylabel('Frequency')
plt.title('Histogram of Errors')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block = True)

plt.figure(figsize=(10, 6))
plt.hist(df_sam_tot['rel_our'], bins = 50,alpha=0.7, label='Our Error')
plt.hist(df_sam_tot['rel_wb'], bins = 50, alpha=0.7, label='WB Error')
plt.xlabel('Error Value')
plt.ylabel('Frequency')
plt.title('Histogram of Errors')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block = True)

import seaborn as sns

df_melted = df_plot.melt(id_vars=['origin', 'destination'], value_vars=['rel_err_our', 'rel_err_wb'],
                    var_name='Error Type', value_name='Error Value')

# Plot with seaborn
plt.figure(figsize=(10, 6))
sns.histplot(data=df_melted, x='Error Value', hue='Error Type', bins=20, kde=True, element='step', stat='density', common_norm=False)
plt.title('Histogram of Errors (Seaborn)')
plt.grid(True)
plt.tight_layout()
plt.show(block = True)

