
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


## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors.pkl"
df = pd.read_pickle(diasporas_file)
df = df.dropna()
df['year'] = df.date.dt.year


##nta accounts
df_nta = pd.read_pickle("C:\\Data\\economic\\nta\\processed_nta.pkl")
nta_dict = {}

###gdp to infer remittances amount
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx").rename(columns = {'country' : 'destination'})#.groupby('country').mean().reset_index().rename(columns = {'country' : 'origin'}).drop(columns = 'year')
df = df.merge(df_gdp, on=['destination', 'year'], how='left')
df['gdp'] = 0.18 * df['gdp'] / 12

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

######## functions
def parse_age_group(age_group_str):
    """Helper function to parse age_group.
       This expects strings like "20-24". """
    lower, upper = map(int, age_group_str.split('-'))
    return lower, upper
def simulate_row_grouped_deterministic(row, separate_disasters=False, group_size=25):
    # Total number of agents for this row
    n_people = int(row['n_people']) // group_size

    # Get lower and upper bounds for the age group.
    lower_age, upper_age = parse_age_group(row['age_group'])

    # Simulate individual ages uniformly within the 5-year range
    # +1 in randint since upper bound is exclusive.
    ages = np.random.randint(lower_age, upper_age + 1, size=n_people)

    # Map the simulated ages to nta values using the dictionary.
    # We assume every age in the simulated sample has an entry in nta_dict.
    nta_values = np.array([nta_dict[age] for age in ages])

    # Simulate years of stay for each agent using the beta parameter.
    yrs_stay = np.random.exponential(scale=row['beta_estimate'], size=n_people).astype(int)

    # Calculate theta for each individual:
    # Here, asymmetry and gdp_diff (and even the beta from the growth rate) are constant for all individuals in the row.
    if separate_disasters:
        theta = constant + (param_nta * (nta_values)) + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['eq_score']) + (row['fl_score']) + (row['st_score']) + (row['dr_score'])
    else:
        theta = constant + (param_nta * (nta_values)) + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['tot_score'])

    # Compute remittance probability using the logistic transformation.
    p = 1 / (1 + np.exp(-theta))
    # p[nta_values == 0] = 0 # Set probability to zero if nta is zero.

    # Simulate the remittance decision (1: sends remittance, 0: does not).
    total_senders = int(sum(p)) * group_size

    # Calculate the total remitted amount for this row.
    # total_remittance = total_senders * fixed_remittance * group_size
    return total_senders

def give_everyone_fixed_probability(row, separate_disasters=False, group_size=25):
    total_senders = int(row['n_people']) * 0.6

    # Calculate the total remitted amount for this row.
    # total_remittance = total_senders * fixed_remittance * group_size
    return total_senders

################### run functions
ita_origin_countries = (df[df.destination == "Italy"]['origin'].unique().tolist())
ita_origin_countries.remove("Cote d'Ivoire")
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

param_nta = 2.5
param_stay = 0
param_asy = -10
param_gdp = 5
# each sender sends 18% of their monthly GDP
# fixed_remittance_ita = df_gdp[df_gdp.destination == 'Italy'].gdp.item() * 0.015 #350  # Amount each sender sends
# fixed_remittance_phil = df_gdp[df_gdp.destination == 'Japan'].gdp.item() * 0.015 #700
# fixed_remittance_mex = df_gdp[df_gdp.destination == 'USA'].gdp.item() * 0.015 #800
# fixed_remittance_nic = df_gdp[df_gdp.destination == 'USA'].gdp.item() * 0.015
# fixed_remittance_gua = df_gdp[df_gdp.destination == 'USA'].gdp.item() * 0.015 #800
# fixed_remittance_pak = df_gdp[df_gdp.destination == 'Italy'].gdp.item() * 0.015
height = 0.6
shape = 1
constant = 0

def plot_country_mean(df, two_countries = False):
    if two_countries:
        df_mean_ita = df[df.destination == 'Italy'][['origin', 'remittances', 'sim_remittances']].groupby(['origin']).mean().reset_index()
        df_mean_phil = df[df.origin == 'Philippines'][['destination', 'remittances', 'sim_remittances']].groupby(['destination']).mean().reset_index()
        df_mean_pak = df[df.origin == 'Pakistan'][['destination', 'remittances', 'sim_remittances']].groupby(
            ['destination']).mean().reset_index()
        df_mean_nic = df[df.origin == 'Nicaragua'][['destination', 'remittances', 'sim_remittances']].groupby(
            ['destination']).mean().reset_index()
        df_mean_mex = df[df.origin.isin(["Mexico", "Guatemala"])][['origin', 'remittances', 'sim_remittances']].groupby(
            ['origin']).mean().reset_index()
        df_mean = pd.concat([df_mean_ita, df_mean_phil, df_mean_mex, df_mean_pak, df_mean_nic])
        fig = go.Figure()

        # Add traces with loop
        for df, color, name, text_col, prefix in zip(
                [df_mean_ita, df_mean_phil, df_mean_mex, df_mean_pak, df_mean_nic],
                ['blue', 'red', 'orange', 'green', 'pink'],
                ['From Italy', 'To Philippines', 'from USA', 'To Pakistan', 'To Nicaragua'],
                ['origin', 'destination', 'origin', 'destination', 'destination'],
                ['Origin', 'Destination', 'origin', 'destination', 'destination']
        ):
            fig.add_trace(go.Scatter(
                x=df['remittances'],
                y=df['sim_remittances'],
                mode='markers',
                name=name,
                marker=dict(color=color, size = 10),
                text=df[text_col],
                hovertemplate=f'{prefix}: %{{text}}<br>Remittances: %{{x}}<br>Simulated: %{{y}}'
            ))

        # Add 1:1 line
        max_val = max(df_mean_ita['remittances'].max(), df_mean_phil['remittances'].max(), df_mean_mex['remittances'].max())
        fig.add_trace(go.Scatter(
            x=np.linspace(0, max_val, 100),
            y=np.linspace(0, max_val, 100),
            mode='lines',
            name='1:1 Line',
            line=dict(color='gray', dash='dash')
        ))

        fig.update_layout(
            title='Simulated vs Actual Remittances',
            xaxis=dict(title='Actual Remittances (log scale)'),
            yaxis=dict(title='Simulated Remittances (log scale)'),
            legend=dict(title='Legend'),
            template='plotly_white'
        )
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    else:
        df_mean = df[['origin', 'remittances', 'sim_remittances']].groupby(['origin']).mean().reset_index()
        fig = px.scatter(df_mean, x = 'remittances', y = 'sim_remittances',
                         color = 'origin', log_x=True, log_y=True)
        fig.add_scatter(x=np.linspace(0, df_mean.remittances.max(), 100),
                        y=np.linspace(0, df_mean.remittances.max(), 100))
    fig.show()
    goodness_of_fit_results(df_mean)

def check_initial_guess_no_disasters(fixed_probability = False, plot = False):
    countries_ita = ita_all_countries
    countries_phil = phil_all_countries
    countries_pak = pak_all_countries
    countries_nic = nic_all_countries
    global nta_dict
    # df country
    df_country_ita = df.query(f"""`origin` in {countries_ita} and `destination` == 'Italy'""")
    df_country_phil = df.query(f"""`origin` == 'Philippines' and `destination` in {countries_phil}""")
    df_country_pak = df.query(f"""`origin` == 'Pakistan' and `destination` in {countries_pak}""")
    df_country_mex = df.query(f"""`origin` == 'Mexico' and `destination` == 'USA'""")
    df_country_nic = df.query(f"""`origin` == 'Nicaragua' and `destination` in {countries_nic}""")
    df_country_gua = df.query(f"""`origin` == 'Guatemala' and `destination` == 'USA'""")

    df_country_ita = df_country_ita[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    df_country_phil = df_country_phil[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    df_country_pak = df_country_pak[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    df_country_nic = df_country_nic[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    df_country_mex = df_country_mex[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    df_country_gua = df_country_gua[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()

    ## nta
    df_nta_ita = df_nta.query(f"""`country` == 'Italy'""")[['age', 'nta']].fillna(0)
    df_nta_phil = df_nta[df_nta.country.isin(countries_phil)][['country', 'age', 'nta']].fillna(0)
    df_nta_pak = df_nta[df_nta.country.isin(countries_pak)][['country', 'age', 'nta']].fillna(0)
    df_nta_nic = df_nta[df_nta.country.isin(countries_nic)][['country', 'age', 'nta']].fillna(0)
    df_nta_mex = df_nta.query(f"""`country` == 'USA'""")[['age', 'nta']].fillna(0)

    ### no climate scores
    df_country_ita['tot_score'] = 0
    df_country_mex['tot_score'] = 0
    df_country_gua['tot_score'] = 0

    if not fixed_probability:
        print("Italy ...")
        for ind, row in df_nta_ita.iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        df_country_ita['sim_senders'] = df_country_ita.apply(simulate_row_grouped_deterministic, axis=1)
        print("Mexico ...")
        for ind, row in df_nta_mex.iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        df_country_mex['sim_senders'] = df_country_mex.apply(simulate_row_grouped_deterministic, axis=1)
        df_country_gua['sim_senders'] = df_country_gua.apply(simulate_row_grouped_deterministic, axis=1)
    else:
        df_country_ita['sim_senders'] = df_country_ita.apply(give_everyone_fixed_probability, axis=1)
        df_country_mex['sim_senders'] = df_country_mex.apply(give_everyone_fixed_probability, axis=1)
        df_country_gua['sim_senders'] = df_country_gua.apply(give_everyone_fixed_probability, axis=1)

    df_country_ita['sim_remittances'] = df_country_ita['sim_senders'] * fixed_remittance_ita
    df_country_mex['sim_remittances'] = df_country_mex['sim_senders'] * fixed_remittance_mex
    df_country_gua['sim_remittances'] = df_country_gua['sim_senders'] * fixed_remittance_gua

    ## PHILIPPINES
    print("Philippines ...")
    list_sims = []
    for country in list(set(df_nta_phil.country).intersection(set(phil_all_countries))):
        df_sim = df_country_phil[df_country_phil.destination == country].copy()
        for ind, row in df_nta_phil[df_nta_phil.country == country].iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        df_sim['tot_score'] = 0
        if not fixed_probability:
            df_sim['sim_senders'] = df_sim.apply(simulate_row_grouped_deterministic, axis=1)
        else:
            df_sim['sim_senders'] = df_sim.apply(give_everyone_fixed_probability, axis=1)
        list_sims.append(df_sim)
    df_country_phil = pd.concat(list_sims)
    df_country_phil['sim_remittances'] = df_country_phil['sim_senders'] * fixed_remittance_phil

    ## PAKISTAN
    print("Pakistan ...")
    list_sims = []
    for country in list(set(df_nta_pak.country).intersection(set(pak_all_countries))):
        df_sim = df_country_pak[df_country_pak.destination == country].copy()
        for ind, row in df_nta_pak[df_nta_pak.country == country].iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        df_sim['tot_score'] = 0
        if not fixed_probability:
            df_sim['sim_senders'] = df_sim.apply(simulate_row_grouped_deterministic, axis=1)
        else:
            df_sim['sim_senders'] = df_sim.apply(give_everyone_fixed_probability, axis=1)
        list_sims.append(df_sim)
    df_country_pak = pd.concat(list_sims)
    df_country_pak['sim_remittances'] = df_country_pak['sim_senders'] * fixed_remittance_pak

    ## NICARAGUA
    print("Nicaragua ...")
    list_sims = []
    for country in list(set(df_nta_nic.country).intersection(set(nic_all_countries))):
        df_sim = df_country_nic[df_country_nic.destination == country].copy()
        for ind, row in df_nta_nic[df_nta_nic.country == country].iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        df_sim['tot_score'] = 0
        if not fixed_probability:
            df_sim['sim_senders'] = df_sim.apply(simulate_row_grouped_deterministic, axis=1)
        else:
            df_sim['sim_senders'] = df_sim.apply(give_everyone_fixed_probability, axis=1)
        list_sims.append(df_sim)
    df_country_nic = pd.concat(list_sims)
    df_country_nic['sim_remittances'] = df_country_nic['sim_senders'] * fixed_remittance_nic

    df_country = pd.concat([df_country_ita, df_country_phil, df_country_mex, df_country_pak, df_country_gua, df_country_nic])
    remittance_per_period = df_country.groupby(['date', 'origin', 'destination'])['sim_remittances'].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")
    remittance_per_period['error_squared'] = (remittance_per_period['remittances'] -
                                              remittance_per_period['sim_remittances']) ** 2
    if plot:
        goodness_of_fit_results(remittance_per_period)

        plot_country_mean(remittance_per_period, two_countries=True)

    return remittance_per_period

# rem_per_period = check_initial_guess_no_disasters(fixed_probability=False, plot = True)

## drop mexico
# rem_per_period = rem_per_period[rem_per_period.origin != "Mexico"]
# goodness_of_fit_results(rem_per_period)
# plot_country_mean(rem_per_period, two_countries=True)

############
# now include disasters
dict_scores = dict(zip([x for x in range(12)],
                       zero_values_before_first_positive_and_after_first_negative([height + shape * np.sin((np.pi / 6) * x) for x in range(1,13)])))
def calculate_tot_score(emdat_ita, height, shape, shift):
    global dict_scores
    dict_scores = dict(zip([x for x in range(12)],
                           zero_values_before_first_positive_and_after_first_negative(
                               [height + shape * np.sin((np.pi / 6) * (x+shift)) for x in range(1, 13)])))
    for x in range(12):
        emdat_ita[f"tot_{x}"] = emdat_ita[[f"eq_{x}",f"dr_{x}",f"fl_{x}",f"st_{x}"]].sum(axis =1) * dict_scores[x]
    emdat_ita["tot_score"] = emdat_ita[[x for x in emdat_ita.columns if "tot" in x]].sum(axis =1)
    return emdat_ita[['date', 'origin', 'tot_score']]

def check_initial_guess_with_disasters(height, shape, shift, fixed_probability = False, plot = False):
    countries_ita = ita_all_countries
    countries_phil = phil_all_countries
    countries_pak = pak_all_countries
    countries_nic = nic_all_countries
    global nta_dict
    # df country
    df_country_ita = df.query(f"""`origin` in {countries_ita} and `destination` == 'Italy'""")
    df_country_phil = df.query(f"""`origin` == 'Philippines' and `destination` in {countries_phil}""")
    df_country_pak = df.query(f"""`origin` == 'Pakistan' and `destination` in {countries_pak}""")
    df_country_mex = df.query(f"""`origin` == 'Mexico' and `destination` == 'USA'""")
    df_country_nic = df.query(f"""`origin` == 'Nicaragua' and `destination` in {countries_nic}""")
    df_country_gua = df.query(f"""`origin` == 'Guatemala' and `destination` == 'USA'""")

    df_country_ita = df_country_ita[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year', 'origin', 'age_group', 'mean_age', 'destination', 'gdp']).mean().reset_index()
    df_country_phil = df_country_phil[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp']).mean().reset_index()
    df_country_pak = df_country_pak[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp']).mean().reset_index()
    df_country_nic = df_country_nic[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp']).mean().reset_index()
    df_country_mex = df_country_mex[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp']).mean().reset_index()
    df_country_gua = df_country_gua[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp']).mean().reset_index()

    ## nta
    df_nta_ita = df_nta.query(f"""`country` == 'Italy'""")[['age', 'nta']].fillna(0)
    df_nta_phil = df_nta[df_nta.country.isin(countries_phil)][['country', 'age', 'nta']].fillna(0)
    df_nta_pak = df_nta[df_nta.country.isin(countries_pak)][['country', 'age', 'nta']].fillna(0)
    df_nta_nic = df_nta[df_nta.country.isin(countries_nic)][['country', 'age', 'nta']].fillna(0)
    df_nta_mex = df_nta.query(f"""`country` == 'USA'""")[['age', 'nta']].fillna(0)

    ### include climate scores
    emdat_ita = emdat[emdat.origin.isin(df_country_ita.origin.unique())].copy()
    emdat_ita = calculate_tot_score(emdat_ita, height, shape, shift)
    df_country_ita = df_country_ita.merge(emdat_ita, on = ['origin', 'date'], how = 'left')

    emdat_mex = emdat[emdat.origin == "Mexico"].copy()
    emdat_mex = calculate_tot_score(emdat_mex, height, shape, shift)
    df_country_mex = df_country_mex.merge(emdat_mex, on=['origin', 'date'], how='left')

    emdat_phil = emdat[emdat.origin == "Philippines"].copy()
    emdat_phil = calculate_tot_score(emdat_phil, height, shape, shift)
    df_country_phil = df_country_phil.merge(emdat_phil, on=['origin', 'date'], how='left')

    emdat_gua = emdat[emdat.origin == "Guatemala"].copy()
    emdat_gua = calculate_tot_score(emdat_gua, height, shape, shift)
    df_country_gua = df_country_gua.merge(emdat_gua, on=['origin', 'date'], how='left')

    emdat_pak = emdat[emdat.origin == "Pakistan"].copy()
    emdat_pak = calculate_tot_score(emdat_pak, height, shape, shift)
    df_country_pak = df_country_pak.merge(emdat_pak, on=['origin', 'date'], how='left')

    emdat_nic = emdat[emdat.origin == "Nicaragua"].copy()
    emdat_nic = calculate_tot_score(emdat_nic, height, shape, shift)
    df_country_nic = df_country_nic.merge(emdat_nic, on=['origin', 'date'], how='left')

    if not fixed_probability:
        # print("Italy ...")
        for ind, row in df_nta_ita.iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        df_country_ita['sim_senders'] = df_country_ita.apply(simulate_row_grouped_deterministic, axis=1)
        # print("Mexico ...")
        for ind, row in df_nta_mex.iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        df_country_mex['sim_senders'] = df_country_mex.apply(simulate_row_grouped_deterministic, axis=1)
        df_country_gua['sim_senders'] = df_country_gua.apply(simulate_row_grouped_deterministic, axis=1)
    else:
        df_country_ita['sim_senders'] = df_country_ita.apply(give_everyone_fixed_probability, axis=1)
        df_country_mex['sim_senders'] = df_country_mex.apply(give_everyone_fixed_probability, axis=1)
        df_country_gua['sim_senders'] = df_country_gua.apply(give_everyone_fixed_probability, axis=1)

    # df_country_ita = df_country_ita.merge(df_gdp, on=['destination', 'year'], how='left')
    # df_country_ita['gdp'] = 0.18 * df_country_ita['gdp'] / 12
    df_country_ita['sim_remittances'] = df_country_ita['sim_senders'] * df_country_ita['gdp']

    # df_country_mex = df_country_mex.merge(df_gdp, on=['destination', 'year'], how='left')
    # df_country_mex['gdp'] = 0.18 * df_country_mex['gdp'] / 12
    df_country_mex['sim_remittances'] = df_country_mex['sim_senders'] * df_country_mex['gdp']

    # df_country_gua = df_country_gua.merge(df_gdp, on=['destination', 'year'], how='left')
    # df_country_gua['gdp'] = 0.18 * df_country_gua['gdp'] / 12
    df_country_gua['sim_remittances'] = df_country_gua['sim_senders'] * df_country_gua['gdp']

    ## PHILIPPINES
    # print("Philippines ...")
    list_sims = []
    for country in list(set(df_nta_phil.country).intersection(set(phil_all_countries))):
        df_sim = df_country_phil[df_country_phil.destination == country].copy()
        for ind, row in df_nta_phil[df_nta_phil.country == country].iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        if not fixed_probability:
            df_sim['sim_senders'] = df_sim.apply(simulate_row_grouped_deterministic, axis=1)
        else:
            df_sim['sim_senders'] = df_sim.apply(give_everyone_fixed_probability, axis=1)
        list_sims.append(df_sim)
    df_country_phil = pd.concat(list_sims)
    # df_country_phil = df_country_phil.merge(df_gdp, on=['destination', 'year'], how='left')
    # df_country_phil['gdp'] = 0.18 * df_country_phil['gdp'] / 12
    df_country_phil['sim_remittances'] = df_country_phil['sim_senders'] * df_country_phil['gdp']

    ## PAKISTAN
    # print("Pakistan ...")
    list_sims = []
    for country in list(set(df_nta_pak.country).intersection(set(pak_all_countries))):
        df_sim = df_country_pak[df_country_pak.destination == country].copy()
        for ind, row in df_nta_pak[df_nta_pak.country == country].iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        if not fixed_probability:
            df_sim['sim_senders'] = df_sim.apply(simulate_row_grouped_deterministic, axis=1)
        else:
            df_sim['sim_senders'] = df_sim.apply(give_everyone_fixed_probability, axis=1)
        list_sims.append(df_sim)
    df_country_pak = pd.concat(list_sims)
    # df_country_pak = df_country_pak.merge(df_gdp, on=['destination', 'year'], how='left')
    # df_country_pak['gdp'] = 0.18 * df_country_pak['gdp'] / 12
    df_country_pak['sim_remittances'] = df_country_pak['sim_senders'] * df_country_pak['gdp']

    ## NICARAGUA
    # print("Nicaragua ...")
    list_sims = []
    for country in list(set(df_nta_nic.country).intersection(set(nic_all_countries))):
        df_sim = df_country_nic[df_country_nic.destination == country].copy()
        for ind, row in df_nta_nic[df_nta_nic.country == country].iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        if not fixed_probability:
            df_sim['sim_senders'] = df_sim.apply(simulate_row_grouped_deterministic, axis=1)
        else:
            df_sim['sim_senders'] = df_sim.apply(give_everyone_fixed_probability, axis=1)
        list_sims.append(df_sim)
    df_country_nic = pd.concat(list_sims)
    # df_country_nic = df_country_nic.merge(df_gdp, on=['destination', 'year'], how='left')
    # df_country_nic['gdp'] = 0.18 * df_country_nic['gdp'] / 12
    df_country_nic['sim_remittances'] = df_country_nic['sim_senders'] * df_country_nic['gdp']

    df_country = pd.concat([df_country_ita, df_country_phil, df_country_mex, df_country_pak, df_country_gua, df_country_nic])
    remittance_per_period = df_country.groupby(['date', 'origin', 'destination'])[['sim_remittances', 'sim_senders']].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")
    # remittance_per_period['error_squared'] = (remittance_per_period['remittances'] -
    #                                           remittance_per_period['sim_remittances']) ** 2
    if plot:
        goodness_of_fit_results(remittance_per_period)

        plot_country_mean(remittance_per_period, two_countries=True)

    return remittance_per_period

rem_per_period = check_initial_guess_with_disasters(height, shape, shift=1, fixed_probability=False, plot = True)

## compare with and without disasters
rem_no_disaster = check_initial_guess_no_disasters(fixed_probability=False, plot = False)

rem_with_disaster = check_initial_guess_with_disasters(height, shape, fixed_probability=False, plot = True)

rem_comp = rem_no_disaster.merge(rem_with_disaster, on = ['date', 'origin', 'destination', 'remittances', 'year'],
                                 suffixes = ('_no', '_with'))

######################################
# run random selections for the optimisation
#######################################
df = df.merge(df_rem, on =['date', 'origin', 'destination'], how = 'left')
df.dropna(inplace = True)

param_nta = 2# uniform(0., 8)
param_asy = -7.5  # uniform(-4, 0)
param_gdp = 10.53  # uniform(0, 8)
height = -0.3  # uniform(0, 1)
shape = 0.58  # uniform(0, 2)
shift = 2
constant = -0.05

random.seed(1234)

results_list = []
extensive_results_list = []
for i in tqdm(range(500)):
    param_nta = uniform(0.5,3)
    param_asy = uniform(-10,-5)
    param_gdp = uniform(7,12)
    height = uniform(-0.5,1)
    shape = uniform(-0.5,1)
    constant = uniform(-2,2)
    shift = uniform(-2,2)

    rem_per_period = check_initial_guess_with_disasters(height, shape, shift, fixed_probability=False, plot = False)
    dict_params = {"nta": param_nta,
                   "asy": param_asy,
                   "stay": param_stay,
                   "gdp": param_gdp,
                   "height": height,
                   "shape": shape,
                   "shift" : shift,
                   "constant": constant}

    for col in dict_params.keys():
        rem_per_period[col] = dict_params[col]
    extensive_results_list.append(rem_per_period)

all_results = pd.concat(extensive_results_list)
all_results.to_pickle('all_results_0307.pkl')

# db = all_results[['nta', 'asy', 'stay', 'gdp', 'height', 'shape','constant', 'error_squared']].groupby(
#     ['nta', 'asy', 'stay', 'gdp', 'height', 'shape','constant']
# ).sum().reset_index().sort_values('error_squared')
#
# db.to_excel("C://Users//Andrea Vismara//Downloads//simulations_errors_2506.xlsx", index = False)
######
# Manual search?
##
param_nta = 1.815131497405294
param_asy = -7.098063764006957
param_gdp = 9.90727371532861
height = -0.04539165647009096
shape = 0.257441199892752
constant = 0.13281665817199664
shift = 0.09119031690093526

rem_per_period = check_initial_guess_with_disasters(height, shape, shift, fixed_probability=False, plot=True)


##########


# 1. Compute difference metrics
rem_comp['diff_abs'] = rem_comp['sim_remittances_with'] - rem_comp['sim_remittances_no']
rem_comp['diff_pct'] = rem_comp['diff_abs'] / rem_comp['sim_remittances_no'] * 100
rem_comp['error_diff'] = rem_comp['error_squared_no'] - rem_comp['error_squared_with']

# 2. Histogram of squared‐error improvement
plt.figure()
plt.hist(rem_comp['error_diff'].dropna(), bins=1000)
plt.title('Improvement in Squared Error (No vs With)')
plt.xlabel('Error Reduction')
plt.ylabel('Frequency')
plt.show(block = True)

# 3. Annual MSE comparison
annual = rem_comp.groupby('year').agg(
    mse_no=('error_squared_no', 'mean'),
    mse_with=('error_squared_with', 'mean')
).reset_index()

plt.figure()
plt.plot(annual['year'], annual['mse_no'], marker='o', label='No Disasters')
plt.plot(annual['year'], annual['mse_with'], marker='o', label='With Disasters')
plt.title('Annual Mean Squared Error Comparison')
plt.xlabel('Year')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show(block = True)

# 4. Time-series comparison for one country‐pair
origin, destination = 'Philippines', 'Italy'  # replace with any pair in your data
pair_df = rem_comp[(rem_comp['origin'] == origin) & (rem_comp['destination'] == destination)].sort_values('date')

plt.figure()
plt.plot(pair_df['date'], pair_df['sim_remittances_no'], marker='o', label='No Disasters')
plt.plot(pair_df['date'], pair_df['sim_remittances_with'], marker='o', label='With Disasters')
plt.plot(pair_df['date'], pair_df['remittances'], marker='o', label='Actual')
plt.title(f'Remittances: {origin} → {destination}')
plt.xlabel('Date')
plt.ylabel('Remittances')
plt.legend()
plt.grid(True)
plt.show(block = True)
