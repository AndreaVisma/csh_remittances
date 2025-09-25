

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
from utils import zero_values_before_first_positive_and_after_first_negative

param_stay = 0

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors_2207_TRAIN.pkl"
df = pd.read_pickle(diasporas_file)
df = df.dropna()
df['year'] = df.date.dt.year
df = df[(df.year == 2019) & (df.date.dt.month == 12)]

df_gdp = pd.read_pickle("c:\\data\\economic\\gdp\\annual_gdp_per_capita_splined.pkl")
df = df.merge(df_gdp, on=['destination', 'date'], how='left')

## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags_NEW.pkl")

####parameters

params = [2.323125219174453, -8.90268722949454, 9.158294373208719,
            0.1788188275520765, 0.21924650014050806,-0.7500114294211869,
            -0.04579122940366171, 0.13737550869522205]
######## functions

param_nta, param_asy, param_gdp, height, shape, shift, constant, rem_pct = params

######## functions
def calculate_tot_score(emdat_ita, height, shape, shift):
    global dict_scores
    dict_scores = dict(zip([x for x in range(12)],
                           zero_values_before_first_positive_and_after_first_negative(
                               [height + shape * np.sin((np.pi / 6) * (x+shift)) for x in range(1, 13)])))
    for x in range(12):
        emdat_ita[f"tot_{x}"] = emdat_ita[[f"eq_{x}",f"dr_{x}",f"fl_{x}",f"st_{x}"]].sum(axis =1) * dict_scores[x]
    emdat_ita["tot_score"] = emdat_ita[[x for x in emdat_ita.columns if "tot" in x]].sum(axis =1)
    return emdat_ita[['date', 'origin', 'tot_score']]

def calculate_tot_score_specific(emdat_ita, height, shape, shift, disaster):
    global dict_scores
    dict_scores = dict(zip([x for x in range(12)],
                           zero_values_before_first_positive_and_after_first_negative(
                               [height + shape * np.sin((np.pi / 6) * (x+shift)) for x in range(1, 13)])))
    for x in range(12):
        emdat_ita[f"tot_{x}"] = emdat_ita[[f"eq_{x}",f"dr_{x}",f"fl_{x}",f"st_{x}"]].sum(axis =1) * dict_scores[x]
    disasters_dict = dict(zip(["Earthquake", "Flood", "Storm", "Drought"], ["eq", "fl", "st", "dr"]))
    dis_name = disasters_dict[disaster]

    emdat_ita[f"{dis_name}_score"] = emdat_ita[[x for x in emdat_ita.columns if dis_name in x]].sum(axis =1)
    return emdat_ita[['date', 'origin', f"{dis_name}_score"]]

def simulate_remittance_probability(df_countries, height, shape, shift, rem_pct, disasters = True, disaster_impact = 0.01):

    if disasters:
        df_countries['tot_score'] = disaster_impact
    else:
        df_countries['tot_score'] = 0

    # df_countries['rem_amount'] = rem_pct * df_countries['gdp'] / 12

    df_countries['theta'] = constant + (param_nta * (df_countries['nta'])) \
                            + (param_asy * df_countries['asymmetry']) + (param_gdp * df_countries['gdp_diff_norm']) \
                            + (df_countries['tot_score'])
    df_countries['probability'] = 1 / (1 + np.exp(-df_countries["theta"]))
    df_countries.loc[df_countries.nta <= 0.01, 'probability'] = 0
    return df_countries

df_prob = simulate_remittance_probability(df, height, shape, shift, rem_pct, disasters = False)

df_prob_dis = simulate_remittance_probability(df, height, shape, shift, rem_pct, disasters = True)

biggest_countries = (df_prob[["origin", "n_people"]].groupby("origin").sum()
                     .reset_index().sort_values("n_people", ascending=False)).head(100)

##########################
##########################
def create_weighted_profile(group, n_bins=100):
    """Create a weighted probability distribution"""
    sorted_group = group.sort_values('probability')
    probabilities = sorted_group['probability'].values
    weights = sorted_group['n_people'].values
    weights = weights / weights.sum()  # Normalize weights

    # Create cumulative distribution
    cum_weights = np.cumsum(weights)

    # Sample quantiles
    quantiles = np.linspace(0, 1, n_bins)
    profile = np.interp(quantiles, cum_weights, probabilities)

    return profile.tolist()

country_profiles = {}
for country, group in tqdm(df_prob.groupby('origin')):
    # Sort by probability and expand based on population
    sorted_group = group.sort_values('probability')
    profile = []

    for _, row in sorted_group.iterrows():
        profile.extend([row['probability']] * int(row['n_people']))

    country_profiles[country] = profile

# Convert dictionary to DataFrame
df_avg_prob = pd.DataFrame([{"country": k, "avg_prob": sum(v) / len(v) if len(v) > 0 else 0} for k, v in country_profiles.items()])
df_avg_prob = df_avg_prob.sort_values("avg_prob", ascending=False)

smaller_country_profiles = {k: random.sample(country_profiles[k], 1000)
                    for k in country_profiles.keys()
                    if len(country_profiles[k]) > 1000}

interest_keys = ["India", "Switzerland", "Haiti"]
dict_to_plot = {k: smaller_country_profiles[k] for k in interest_keys}

biggest_countries_dict = {k: smaller_country_profiles[k] for k in biggest_countries.origin.tolist()}

fig, ax = plt.subplots(figsize=(7.2, 4.2))

for label, probs in tqdm(biggest_countries_dict.items()):
    probs = np.sort(probs)
    x = np.linspace(0, 1, len(probs))
    plt.plot(x, probs, linewidth=2, color = 'grey', alpha = 0.2)

for label, probs in dict_to_plot.items():
    probs = np.sort(probs)
    x = np.linspace(0, 1, len(probs))
    auc = np.trapz(probs, x)
    plt.plot(x, probs, label=f"{label}, AUC: {round(auc, 3)}", linewidth=3.5)

plt.grid(False)
# plt.legend()
plt.xticks([])
plt.yticks(fontsize=16)
import matplotlib.ticker as mtick
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig('C:\\Users\\Andrea Vismara\\Desktop\\papers\\remittances\\figures\\diaspora_profiles_2015.svg', bbox_inches = 'tight')
plt.show(block = True)

######################################
country_profiles = {}
for country, group in tqdm(df_prob_dis.groupby('origin')):
    # Sort by probability and expand based on population
    sorted_group = group.sort_values('probability')
    profile = []

    for _, row in sorted_group.iterrows():
        profile.extend([row['probability']] * int(row['n_people']))

    country_profiles[country] = profile

smaller_country_profiles = {k: random.sample(country_profiles[k], 1000)
                    for k in country_profiles.keys()
                    if len(country_profiles[k]) > 1000}

interest_keys = ["Nicaragua", "Pakistan", "Mexico", "USA"]
dict_to_plot = {k: smaller_country_profiles[k] for k in interest_keys}

biggest_countries_dict = {k: smaller_country_profiles[k] for k in biggest_countries.origin.tolist()}

fig, ax = plt.subplots(figsize=(5, 9))

for label, probs in tqdm(biggest_countries_dict.items()):
    probs = np.sort(probs)
    x = np.linspace(0, 1, len(probs))
    plt.plot(x, probs, linewidth=2.5, color = 'grey', alpha = 0.3)

for label, probs in dict_to_plot.items():
    probs = np.sort(probs)
    x = np.linspace(0, 1, len(probs))
    auc = np.trapz(probs, x)
    plt.plot(x, probs, label=f"{label}, AUC: {round(auc, 3)}", linewidth=3.5)

plt.grid(False)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
import matplotlib.ticker as mtick
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig('C:\\Users\\Andrea Vismara\\Desktop\\papers\\remittances\\figures\\diaspora_profiles_DISASTERS.png', bbox_inches = 'tight')
plt.show(block = True)

##########################
# country to country
bilateral_profiles = {}


for country, group in tqdm(df_prob.groupby(['origin', 'destination'])):
    # Sort by probability and expand based on population
    sorted_group = group.sort_values('probability')
    profile = []

    for _, row in sorted_group.iterrows():
        profile.extend([row['probability']] * int(row['n_people']))

    bilateral_profiles[str(country)] = profile

smaller_bilateral_profiles = {k: random.sample(bilateral_profiles[k], 10_000)
                    for k in bilateral_profiles.keys()
                    if len(bilateral_profiles[k]) > 300_000}

interest_keys = ["('Bangladesh', 'Saudi Arabia')", "('Guatemala', 'USA')"]
dict_to_plot = {k: smaller_bilateral_profiles[k] for k in interest_keys}

fig, ax = plt.subplots(figsize=(7.4, 4.2))

for label, probs in tqdm(smaller_bilateral_profiles.items()):
    probs = np.sort(probs)
    x = np.linspace(0, 1, len(probs))
    plt.plot(x, probs, linewidth=2, color = 'grey', alpha = 0.2)

for label, probs in dict_to_plot.items():
    probs = np.sort(probs)
    x = np.linspace(0, 1, len(probs))
    auc = np.trapz(probs, x)
    plt.plot(x, probs, label=f"{label}, AUC: {round(auc, 3)}", linewidth=3.5)

plt.grid(False)
# plt.legend()
plt.xticks([])
plt.yticks(fontsize=18)
import matplotlib.ticker as mtick
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
fig.savefig('C:\\Users\\Andrea Vismara\\Desktop\\papers\\remittances\\figures\\diaspora_profiles_BILATERAL.png', bbox_inches = 'tight')
plt.show(block = True)

##########################
## MEXICO profile over time
df_mex  = pd.read_pickle(diasporas_file)
df_mex = df_mex.dropna()
df_mex['year'] = df_mex.date.dt.year
df_mex = df_mex[(df_mex.origin == "Mexico") & (df_mex.date.dt.month == 1) & (df_mex.destination == "USA")].copy()

df_prob_mex = simulate_remittance_probability(df_mex, height, shape, shift, rem_pct, disasters = False)
###
country_profiles = {}
for country, group in tqdm(df_prob_mex.groupby('year')):
    # Sort by probability and expand based on population
    sorted_group = group.sort_values('probability')
    profile = []

    for _, row in sorted_group.iterrows():
        profile.extend([row['probability']] * int(row['n_people']))

    country_profiles[country] = profile

smaller_mex_profiles = {k: random.sample(country_profiles[k], 10_000)
                    for k in country_profiles.keys()
                    if len(country_profiles[k]) > 1000}

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import matplotlib.colors as mcolors

fig, ax = plt.subplots(figsize=(7.4, 4.2))

# Define a colormap (choose one you like, e.g. viridis, plasma, inferno, cividis, etc.)
cmap = plt.cm.Blues
# Normalize years between min and max
years = sorted([int(label) for label in smaller_mex_profiles.keys()])
norm = mcolors.Normalize(vmin=min(years), vmax=max(years))

for label, probs in smaller_mex_profiles.items():
    year = int(label)
    probs = np.sort(probs)
    x = np.linspace(0, 1, len(probs))
    auc = np.trapz(probs, x)

    # Map year to a color
    color = cmap(norm(year))

    ax.plot(
        x, probs,
        label=f"{label}, AUC: {round(auc, 3)}",
        linewidth=3.5,
        color=color
    )
# Formatting
plt.grid(False)
# plt.legend()
plt.xticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.yticks(fontsize=16)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

fig.savefig(
    'C:\\Users\\Andrea Vismara\\Desktop\\papers\\remittances\\figures\\diaspora_profiles_MEXICO.svg',
    bbox_inches='tight'
)
plt.show(block=True)

fig, ax = plt.subplots(figsize=(4, 1.5))

# Create a vertical gradient (100 x 1 array)
gradient = np.linspace(0, 1, 256).reshape(1, -1)

# Show it using the Greens colormap
ax.imshow(gradient, aspect='auto', cmap='Blues')

# Remove axes
ax.set_axis_off()
fig.savefig(
    'C:\\Users\\Andrea Vismara\\Desktop\\papers\\remittances\\figures\\Blues.svg',
    bbox_inches='tight'
)
plt.show(block = True)

##########################
## Income group
from utils import dict_names
income_class = pd.read_excel("C:\\Data\\economic\\income_classification_countries_wb.xlsx")
income_class['country'] = income_class.country.map(dict_names)
income_class = income_class[["country", "group", "Region"]]

interest_keys = income_class[income_class['group'] == "High income"].country.tolist()
high_income_plot = {k: country_profiles[k] for k in interest_keys if k in df_prob.origin.unique()}

biggest_countries_dict = {k: country_profiles[k] for k in biggest_countries.origin.tolist()}

fig, ax = plt.subplots(figsize=(7.5, 6))

for label, probs in tqdm(biggest_countries_dict.items()):
    probs = np.sort(probs)
    x = np.linspace(0, 1, len(probs))
    plt.plot(x, probs, linewidth=2, color = 'grey', alpha = 0.2)

for label, probs in high_income_plot.items():
    probs = np.sort(probs)
    x = np.linspace(0, 1, len(probs))
    auc = np.trapz(probs, x)
    plt.plot(x, probs, label=f"{label}, AUC: {round(auc, 3)}", linewidth=3)

plt.title("Normalized Remittance Probability by Individual")
plt.xlabel("Normalized Individual Position (0 = first, 1 = last)")
plt.ylabel("Probability of Sending Remittances")
# plt.legend()
plt.grid(True)
fig.savefig('.\plots\\for_paper\\diaspora_profiles_2020.png', bbox_inches = 'tight')
plt.show(block = True)

##########################
# Create figure
fig = go.Figure()

# Add the "background" grey lines (results dict)
for label, probs in tqdm(smaller_bilateral_profiles.items()):
    probs = np.sort(probs)
    x = np.linspace(0, 1, len(probs))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=probs,
            mode="lines",
            line=dict(width=2, color="grey"),
            opacity=0.2,
            name=label,  # if you want every line in legend, remove 'showlegend=False'
            showlegend=False  # avoids cluttering legend with all results
        )
    )

# Add highlighted lines (dict_to_plot)
for label, probs in dict_to_plot.items():
    probs = np.sort(probs)
    x = np.linspace(0, 1, len(probs))
    auc = np.trapz(probs, x)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=probs,
            mode="lines",
            line=dict(width=3),
            name=f"{label}, AUC: {round(auc, 3)}",
            hovertemplate="Position: %{x:.2f}<br>Probability: %{y:.3f}<extra>"+label+"</extra>"
        )
    )

# Layout formatting
fig.update_layout(
    title="Normalized Remittance Probability by Individual",
    xaxis_title="Normalized Individual Position (0 = first, 1 = last)",
    yaxis_title="Probability of Sending Remittances",
    legend_title="Country Profiles",
    width=750,
    height=600,
    template="plotly_white"
)

fig.show()

##########################
results = {}

k = 10_000
for (origin, destination), df_sub in df_prob.groupby(["origin", "destination"]):
    colors = df_sub["n_people"].to_numpy(dtype=np.int64)   # counts per unique prob
    probs = df_sub["probability"].to_numpy()              # corresponding probability values
    total = int(colors.sum())
    if total < 10_000:
        results[(origin, destination)] = []
        continue

    nsample = min(k, total)
    draws = rng.multivariate_hypergeometric(colors, nsample)  # fast; draws.sum() == nsample

    sampled_probs = np.repeat(probs, draws)
    results[(origin, destination)] = sampled_probs.tolist()  # <= 10k elements

# Now specify which pairs to highlight:
interest_keys = [
    ("Nicaragua", "USA"),
    ("Pakistan", "Saudi Arabia"),
    ("India", "UAE"),
    ("Switzerland", "Germany"),
]

dict_to_plot = {k: results[k] for k in interest_keys if k in results}

# Create figure
fig = go.Figure()
# Add the "background" grey lines (results dict)
# for label, probs in tqdm(results.items()):
#     probs = np.sort(probs)
#     x = np.linspace(0, 1, len(probs))
#     fig.add_trace(
#         go.Scatter(
#             x=x,
#             y=probs,
#             mode="lines",
#             line=dict(width=2, color="grey"),
#             opacity=0.2,
#             name=str(label),  # if you want every line in legend, remove 'showlegend=False'
#             showlegend=False  # avoids cluttering legend with all results
#         )
#     )

# Add highlighted lines (dict_to_plot)
for label, probs in dict_to_plot.items():
    probs = np.sort(probs)
    x = np.linspace(0, 1, len(probs))
    auc = np.trapz(probs, x)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=probs,
            mode="lines",
            line=dict(width=3),
            name=f"{label}, AUC: {round(auc, 3)}",
            hovertemplate="Position: %{x:.2f}<br>Probability: %{y:.3f}<extra>"+str(label)+"</extra>"
        )
    )

# Layout formatting
fig.update_layout(
    title="Normalized Remittance Probability by Individual",
    xaxis_title="Normalized Individual Position (0 = first, 1 = last)",
    yaxis_title="Probability of Sending Remittances",
    legend_title="Country Profiles",
    width=750,
    height=600,
    template="plotly_white"
)

fig.show()
##########################
