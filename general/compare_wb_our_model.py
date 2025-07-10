
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

beta = 0.7
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

param_stay = 0

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors.pkl"
df = pd.read_pickle(diasporas_file)
df = df[df.origin != "Libya"]
df = df.dropna()
df['year'] = df.date.dt.year

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
df['rem_amount'] = 0.18 * df['gdp'] / 12

## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags.pkl")

## load italy & Philippines remittances
#ITA
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
df_rem.sort_values(['origin', 'date'], inplace=True)
df_rem_group = df_rem.copy()
df_rem_group['year'] = df_rem_group["date"].dt.year

df = df.merge(df_rem, on =['date', 'origin', 'destination'], how = 'left')
df.dropna(inplace = True)
unique_pairs = df[['origin', 'destination']].drop_duplicates()
sampled_pairs = unique_pairs.sample(frac=0.6, random_state=42)
df_sampled = df.merge(sampled_pairs, on=['origin', 'destination'], how='inner')

df_saved = df.copy()
not_sampled_pairs = unique_pairs.merge(sampled_pairs, on=['origin', 'destination'], how='outer', indicator=True)
not_sampled_pairs = not_sampled_pairs[not_sampled_pairs['_merge'] == 'left_only'].drop(columns=['_merge'])
df_not_sampled = df_saved.merge(not_sampled_pairs, on=['origin', 'destination'], how='inner')

######## functions

def simulate_row_grouped_deterministic(row, separate_disasters=False):
    # Total number of agents for this row
    n_people = row['n_people']

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

    total_senders = int(p * n_people)

    return total_senders

################### run functions
def get_df_countries(df):
    ita_origin_countries = (df[df.destination == "Italy"]['origin'].unique().tolist())
    try:
        ita_origin_countries.remove("Cote d'Ivoire")
    except:
        pass
    ita_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.destination == "Italy")].origin.unique().tolist()
    ita_all_countries = list(set(ita_origin_countries).intersection(set(ita_countries_high_remittances)))

    phil_dest_countries = (df[df.origin == "Philippines"]['destination'].unique().tolist())
    phil_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.origin == "Philippines")].destination.unique().tolist()
    phil_all_countries = list(set(phil_dest_countries).intersection(set(phil_countries_high_remittances)))

    pak_dest_countries = (df[df.origin == "Pakistan"]['destination'].unique().tolist())
    pak_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.origin == "Pakistan")].destination.unique().tolist()
    pak_all_countries = list(set(pak_dest_countries).intersection(set(pak_countries_high_remittances)))

    nic_dest_countries = (df[df.origin == "Nicaragua"]['destination'].unique().tolist())
    nic_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.origin == "Nicaragua")].destination.unique().tolist()
    nic_all_countries = list(set(nic_dest_countries).intersection(set(nic_countries_high_remittances)))

    countries_ita = ita_all_countries
    countries_phil = phil_all_countries
    countries_pak = pak_all_countries
    countries_nic = nic_all_countries

    # df country
    df_country_ita = df.query(f"""`origin` in {countries_ita} and `destination` == 'Italy'""")
    df_country_phil = df.query(f"""`origin` == 'Philippines' and `destination` in {countries_phil}""")
    df_country_pak = df.query(f"""`origin` == 'Pakistan' and `destination` in {countries_pak}""")
    df_country_mex = df.query(f"""`origin` == 'Mexico' and `destination` == 'USA'""")
    df_country_nic = df.query(f"""`origin` == 'Nicaragua' and `destination` in {countries_nic}""")
    df_country_gua = df.query(f"""`origin` == 'Guatemala' and `destination` == 'USA'""")

    df_country_ita = df_country_ita[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year', 'origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_phil = df_country_phil[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_pak = df_country_pak[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_nic = df_country_nic[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_mex = df_country_mex[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_gua = df_country_gua[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()

    df_countries = pd.concat([df_country_ita, df_country_mex, df_country_gua, df_country_nic, df_country_pak, df_country_phil])
    return df_countries


def return_train_test_result(params):
    global param_nta, param_asy, param_gdp, height, shape, shift, constant, rem_pct
    param_nta, param_asy, param_gdp, height, shape, shift, constant, rem_pct = params

    df_countries = get_df_countries(df_sampled)
    emdat_ita = emdat[emdat.origin.isin(df_countries.origin.unique())].copy()
    emdat_ita = calculate_tot_score(emdat_ita, height, shape, shift)
    try:
        df_countries.drop(columns = 'tot_score', inplace = True)
    except:
        pass
    df_countries = df_countries.merge(emdat_ita, on=['origin', 'date'], how='left')
    df_countries['rem_amount'] = rem_pct * df_countries['gdp'] / 12

    df_countries['sim_senders'] = df_countries.apply(simulate_row_grouped_deterministic, axis=1)
    df_countries['sim_remittances'] = df_countries['sim_senders'] * df_countries['rem_amount']

    remittance_per_period_1 = df_countries.groupby(['date', 'origin', 'destination'])[['sim_remittances', 'sim_senders']].sum().reset_index()
    remittance_per_period_1 = remittance_per_period_1.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")
    remittance_per_period_1['type'] = 'train'

    df_countries = get_df_countries(df_not_sampled)
    emdat_ita = emdat[emdat.origin.isin(df_countries.origin.unique())].copy()
    emdat_ita = calculate_tot_score(emdat_ita, height, shape, shift)
    try:
        df_countries.drop(columns = 'tot_score', inplace = True)
    except:
        pass
    df_countries = df_countries.merge(emdat_ita, on=['origin', 'date'], how='left')
    df_countries['rem_amount'] = rem_pct * df_countries['gdp'] / 12

    df_countries['sim_senders'] = df_countries.apply(simulate_row_grouped_deterministic, axis=1)
    df_countries['sim_remittances'] = df_countries['sim_senders'] * df_countries['rem_amount']

    remittance_per_period = df_countries.groupby(['date', 'origin', 'destination'])[['sim_remittances', 'sim_senders']].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")
    remittance_per_period['type'] = 'test'

    return pd.concat([remittance_per_period_1, remittance_per_period])

def plot_train_test(df_test):

    train = df_test[df_test.type == 'train']
    test = df_test[df_test.type == 'test']

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(train['remittances'], train['sim_remittances'], alpha=0.5, label = 'Training sample')
    ax.scatter(test['remittances'], test['sim_remittances'], alpha=0.5, label='Test sample', marker = 'x')
    lims = [0, train['remittances'].max()]
    ax.plot(lims, lims, 'k-', alpha=1, zorder=1)
    plt.xlabel('Observed Remittances')
    plt.ylabel('Simulated Remittances')
    plt.title("Calibration results")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    # Training sample plot
    axs[0].scatter(train['remittances'], train['sim_remittances'], alpha=0.5, label='Training sample')
    lims = [0, max(train['remittances'].max(), train['sim_remittances'].max())]
    axs[0].plot(lims, lims, 'k-', alpha=1, zorder=1)
    axs[0].set_title("Training Sample")
    axs[0].set_xlabel("Observed Remittances")
    axs[0].set_ylabel("Simulated Remittances")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].grid(True)

    # Test sample plot
    axs[1].scatter(test['remittances'], test['sim_remittances'], alpha=0.5, label='Test sample', marker='x', color = 'orange')
    lims = [0, max(test['remittances'].max(), test['sim_remittances'].max())]
    axs[1].plot(lims, lims, 'k-', alpha=1, zorder=1)
    axs[1].set_title("Test Sample")
    axs[1].set_xlabel("Observed Remittances")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle("Calibration Results")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block = True)

df_test = return_train_test_result([2.66093400319252, -9.771767189034518, 8.310441847102053,
                                    0.338599981797477, 0.3901969522813624,-0.750105008323315,
                                    0.6917237435288245, 0.1942257641159886])

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# Training sample plot
df_test['error'] = df_test['remittances'] - df_test['sim_remittances']
SS_res_1 = np.sum(np.square(df_test['error']))
SS_tot_1 = np.sum(np.square(df_test['remittances'] - np.mean(df_test['remittances'])))
R_squared_1 = round(1 - (SS_res_1 / SS_tot_1),2)

axs[0].scatter(df_test['remittances'], df_test['sim_remittances'], alpha=0.7, label=f'Our Model\n R^2: {R_squared_1}')
lims = [0, max(df_test['remittances'].max(), df_test['sim_remittances'].max())]
axs[0].plot(lims, lims, 'k-', alpha=1, zorder=1)
axs[0].set_title("Our model")
axs[0].set_xlabel("Observed Remittances")
axs[0].set_ylabel("Simulated Remittances")
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].legend()
axs[0].grid(True)

# Test sample plot
df_wb['error'] = df_wb['remittances'] - df_wb['sim_remittances']
SS_res_2 = np.sum(np.square(df_wb['error']))
SS_tot_2 = np.sum(np.square(df_wb['remittances'] - np.mean(df_wb['remittances'])))
R_squared_2 = round(1 - (SS_res_2 / SS_tot_2),2)

axs[1].scatter(df_wb['remittances'], df_wb['sim_remittances'], alpha=0.5, label=f'WB model\n R^2: {R_squared_2}', marker='x', color='green')
lims = [0, max(df_wb['remittances'].max(), df_wb['sim_remittances'].max())]
axs[1].plot(lims, lims, 'k-', alpha=1, zorder=1)
axs[1].set_title("World Bank model")
axs[1].set_xlabel("Observed Remittances")
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].legend()
axs[1].grid(True)

plt.suptitle("Models comparison")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show(block=True)

