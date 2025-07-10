

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
from random import sample, uniform
import random
from pandas.tseries.offsets import MonthEnd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)
from utils import zero_values_before_first_positive_and_after_first_negative

param_stay = 0

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors.pkl"
df = pd.read_pickle(diasporas_file)
df = df[df.origin != "Libya"]
df = df.dropna()
df['year'] = df.date.dt.year
df = df[(df.year == 2020) & (df.date.dt.month == 1)]

##nta accounts
df_nta = pd.read_pickle("C:\\Data\\economic\\nta\\processed_nta.pkl")
nta_dict = {}

df['mean_age'] = df['mean_age'].astype(int)
for country in tqdm(df.destination.unique()):
    for ind, row in df_nta[df_nta.country == country].iterrows():
        nta_dict[int(row.age)] =row.nta
    df.loc[df.destination == country, 'nta'] = df.loc[df.destination == country, 'mean_age'].map(nta_dict)

###gdp to infer remittances amount
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx").rename(columns = {'country' : 'destination'})#.groupby('country').mean().reset_index().rename(columns = {'country' : 'origin'}).drop(columns = 'year')
df = df.merge(df_gdp, on=['destination', 'year'], how='left')

######## functions

def simulate_row_probability(row, separate_disasters=False):
    n_people = int(row['n_people'])

    if row["nta"] != 0:
        if separate_disasters:
            theta = constant + (param_nta * (row['nta'])) \
                    + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                    + (row['eq_score']) + (row['fl_score']) + (row['st_score']) + (row['dr_score'])
        else:
            theta = constant + (param_nta * (row['nta'])) \
                    + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                    + (row['tot_score'])
        # Compute remittance probability using the logistic transformation.
        p = 1 / (1 + np.exp(-theta))
    else:
        p = 0


    return [p] * n_people

def calculate_tot_score(emdat_ita, height, shape, shift):
    global dict_scores
    dict_scores = dict(zip([x for x in range(12)],
                           zero_values_before_first_positive_and_after_first_negative(
                               [height + shape * np.sin((np.pi / 6) * (x+shift)) for x in range(1, 13)])))
    for x in range(12):
        emdat_ita[f"tot_{x}"] = emdat_ita[[f"eq_{x}",f"dr_{x}",f"fl_{x}",f"st_{x}"]].sum(axis =1) * dict_scores[x]
    emdat_ita["tot_score"] = emdat_ita[[x for x in emdat_ita.columns if "tot" in x]].sum(axis =1)
    return emdat_ita[['date', 'origin', 'tot_score']]

def produce_diaspora_profile(df_countries, disasters = False, disaster_size = 1):
    if disasters:
        df_countries['tot_score'] = disaster_size / 100
    else:
        df_countries['tot_score'] = 0

    df_countries['probability'] = df_countries.apply(simulate_row_probability, axis=1)

    return df_countries

params = [2.66093400319252, -9.771767189034518, 8.310441847102053,
                                    0.338599981797477, 0.3901969522813624,-0.750105008323315,
                                    0.6917237435288245, 0.1942257641159886]
param_nta, param_asy, param_gdp, height, shape, shift, constant, pct_rem = params
df_prob = produce_diaspora_profile(df, disasters=False)

results = {}
for country in tqdm(df_prob.destination.unique()):
    df_sub = df_prob[df_prob.destination == country].copy()
    for or_ in df_sub.origin.unique():
        group = (or_, country)
        prob_list = df_sub[df_sub.origin == or_]['probability'].explode().tolist()
        results[group] = prob_list

def sample_from_dict(d, sample=10):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))

# dict_to_plot = sample_from_dict(results, 4)
interest_keys = [('Switzerland', 'Italy'), ('Guatemala', 'USA'), ('Philippines', 'Italy'), ('Pakistan', 'United Kingdom'),
                 ('Mexico', 'USA'), ('Nicaragua', 'Costa Rica'), ('Pakistan', 'Sweden'), ('Germany', 'Italy'), ('Pakistan', 'Saudi Arabia')]
# interest_keys = [('Mexico', 'USA'), ('Nicaragua', 'Costa Rica'), ('Pakistan', 'Sweden'), ('Philippines', 'Italy')]
dict_to_plot = {k: results[k] for k in interest_keys}

fig, ax = plt.subplots(figsize=(7.5, 6))

# for label, probs in tqdm(results.items()):
#     if len(probs) > 10_000:
#         sampled_probs = random.sample([x for x in probs if not np.isnan(x)],10_000)
#     else:
#         sampled_probs = [x for x in probs if not np.isnan(x)]
#     sampled_probs = np.sort(sampled_probs)
#     x = np.linspace(0, 1, len(sampled_probs))
#     if len(sampled_probs) > 100:
#         plt.plot(x, sampled_probs, linewidth=2, color = 'grey', alpha = 0.2)

for label, probs in dict_to_plot.items():
    sampled_probs = [x for x in probs if not np.isnan(x)]
    sampled_probs = np.sort(sampled_probs)
    x = np.linspace(0, 1, len(sampled_probs))
    auc = np.trapz(sampled_probs, x)
    plt.plot(x, sampled_probs, label=f"{label}, AUC: {round(auc, 3)}", linewidth=3)

plt.title("Normalized Remittance Probability by Individual")
plt.xlabel("Normalized Individual Position (0 = first, 1 = last)")
plt.ylabel("Probability of Sending Remittances")
plt.legend()
plt.grid(True)
# fig.savefig('.\plots\\for_paper\\diaspora_profiles_2018.png', bbox_inches = 'tight')
plt.show(block = True)


################################
# compare with and without disasters
interest_keys = [('Switzerland', 'Italy'), ('Mexico', 'USA'), ('Nicaragua', 'Costa Rica'), ('Pakistan', 'Saudi Arabia')]

df_prob_no = produce_diaspora_profile(disasters=False, disaster_size=1)
results_no = {}
for country in tqdm(df_prob.destination.unique()):
    df_sub = df_prob_no[df_prob_no.destination == country].copy()
    for or_ in df_sub.origin.unique():
        group = (or_, country)
        prob_list = df_sub[df_sub.origin == or_]['probability'].explode().tolist()
        results_no[group] = prob_list
dict_to_plot_no = {k: results_no[k] for k in interest_keys}

df_prob_with = produce_diaspora_profile(disasters=True, disaster_size=1)
results_with = {}
for country in tqdm(df_prob.destination.unique()):
    df_sub = df_prob_with[df_prob_with.destination == country].copy()
    for or_ in df_sub.origin.unique():
        group = (or_, country)
        prob_list = df_sub[df_sub.origin == or_]['probability'].explode().tolist()
        results_with[group] = prob_list
dict_to_plot_with = {k: results_with[k] for k in interest_keys}

fig, ax = plt.subplots(figsize = (7.5,9))
aucs_no = []
# colors = []
# for label, probs in dict_to_plot_no.items():
#     sampled_probs = [x for x in probs if not np.isnan(x)]
#     sampled_probs = np.sort(sampled_probs)
#     x = np.linspace(0, 1, len(sampled_probs))
#     auc = np.trapz(sampled_probs, x)
#     plt.plot(x, sampled_probs, label=f"No dis: {label}, AUC: {round(auc, 3)}", linewidth=3)
#     aucs_no.append(auc)
aucs_with = []
for label, probs in dict_to_plot_with.items():
    sampled_probs = [x for x in probs if not np.isnan(x)]
    sampled_probs = np.sort(sampled_probs)
    x = np.linspace(0, 1, len(sampled_probs))
    auc = np.trapz(sampled_probs, x)
    ax.plot(x, sampled_probs, label=f"With dis: {label}, AUC: {round(auc, 3)}", linewidth=3)
    aucs_with.append(auc)

difference = [aucs_with[i] - aucs_no[i] for i in range(len(aucs_no))]

plt.title("Normalized Remittance Probability by Individual")
plt.xlabel("Normalized Individual Position (0 = first, 1 = last)")
plt.ylabel("Probability of Sending Remittances")
# plt.legend()
plt.grid(True)
fig.savefig('.\plots\\for_paper\\diaspora_profiles_2018_w_dis.png', bbox_inches = 'tight')
plt.show(block = True)