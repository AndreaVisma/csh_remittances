
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time
import itertools
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from random import sample
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)

## pair of countries
origin, destination = "Philippines", "Japan"

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_interpolated.pkl"
df = pd.read_pickle(diasporas_file)
df = df[df.n_people > 0]

##exponential betas for years of stay
df_betas = pd.read_pickle("C:\\Data\\migration\\simulations\\exponential_betas.pkl")

## family asymmetry
asymmetry_file = "C:\\Data\\migration\\bilateral_stocks\\pyramid_asymmetry_beginning_of_the_year.pkl"
asy_df = pd.read_pickle(asymmetry_file)

## diaspora growth rates
growth_rates = pd.read_pickle("C://data//migration//stock_pct_change.pkl")

## gdp differential
df_gdp = (pd.read_pickle("c:\\data\\economic\\gdp\\annual_gdp_deltas.pkl"))

##nta accounts
df_nta = pd.read_pickle("C:\\Data\\economic\\nta\\processed_nta.pkl")
nta_dict = {}

## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags_NEW.pkl")


#########################################
#########################################
# Sample parameters
param_nta = 1
param_stay = -0.2
param_asy = -3.5
param_gdp = 0.5
fixed_remittance = 1100  # Amount each sender sends

## load italy remittances
df_rem = pd.read_parquet("C:\\Data\\my_datasets\\italy\\simulation_data.parquet")
df_rem['date'] = pd.to_datetime(df_rem.date)
df_rem.sort_values(['country', 'date'], inplace=True)
df_rem_group = df_rem[~df_rem[["date", "country"]].duplicated()][
    ["date", "country", "remittances", "gdp_per_capita", "delta_gdp"]]
df_pop_group = df_rem[["date", "country", "population"]].groupby(["date", "country"]).sum().reset_index()
df_rem_group = df_rem_group.merge(df_pop_group, on=["date", "country"], how='left')
df_rem_group['exp_pop'] = df_rem_group['remittances'] / fixed_remittance
df_rem_group['pct_sending'] = df_rem_group['exp_pop'] / df_rem_group['population']
df_rem_group['year'] = df_rem_group["date"].dt.year
df_rem_group.rename(columns = {"country" : "origin"}, inplace = True)
df_rem_group = df_rem_group[df_rem_group.columns[:3]]

##### disasters parameters

dis_params = pd.read_excel("C:\\Data\\my_datasets\\disasters\\disasters_params.xlsx", sheet_name="Sheet2").dropna()


def sin_function(a,b,c,x):
    return a * np.sin((np.pi/6) * x) + b * np.sin((np.pi/3) * x) + c

def sin_function_simple(a,c,x):
    return a + c * np.sin((np.pi/6) * x)


def zero_values_before_first_positive_and_after_first_negative(lst):
    modified = lst.copy()
    # Find first positive
    first_positive = next((i for i, x in enumerate(lst) if x > 0), None)

    if first_positive is not None:
        # Zero before first positive
        for i in range(first_positive):
            modified[i] = 0

        # Find first negative AFTER the first positive
        first_negative_after = next(
            (i for i, x in enumerate(lst[first_positive:], start=first_positive) if x < 0),
            None
        )

        if first_negative_after is not None:
            # Zero positives after first negative encountered post-positive
            for i in range(first_negative_after, len(modified)):
                if modified[i] > 0:
                    modified[i] = 0

    return modified
def disaster_score_function(disasters = ['tot'], simple = True):
    global dict_dis_par
    dict_dis_par = {}
    for dis in disasters:
        if not simple:
            a,b,c = dis_params[dis]
            values = [sin_function(a,b,c,x) for x in np.linspace(0, 11, 12)]
            values = zero_values_before_first_positive_and_after_first_negative(values.copy())
        else:
            a,c = dis_params[dis]
            values = [sin_function_simple(a,c,x) for x in np.linspace(0, 11, 12)]
            values = zero_values_before_first_positive_and_after_first_negative(values.copy())
        dict_dis_par[dis] = values
    return dict_dis_par

dict_dis_par = disaster_score_function(disasters = ['tot'], simple=True)

####################
def compute_disasters_scores(df, dict_dis_par):
    df_dis = df.copy()
    df_dis = df_dis.drop_duplicates(subset=["date", "origin"])
    for col in ['eq', 'dr', 'fl', 'st']:
        for shift in [int(x) for x in np.linspace(1, 11, 11)]:
            g = df_dis.groupby('origin', group_keys=False)
            g = g.apply(lambda x: x.set_index(['date', 'origin'])[col]
                        .shift(shift).reset_index(drop=True)).fillna(0)
            df_dis[f'{col}_{shift}'] = g.iloc[0]
            df_dis['tot'] = df_dis['fl'] + df_dis['eq'] + df_dis['st'] + df_dis['dr']
    for shift in [int(x) for x in np.linspace(1, 11, 11)]:
        df_dis[f'tot_{shift}'] = df_dis[f'fl_{shift}'] + df_dis[f'eq_{shift}'] + df_dis[f'st_{shift}'] + df_dis[
            f'dr_{shift}']
    df_dis.rename(columns={'eq': 'eq_0', 'st': 'st_0', 'fl': 'fl_0', 'dr': 'dr_0', 'tot': 'tot_0'}, inplace=True)
    required_columns = ['date', 'origin'] + \
                       [f"{disaster}_{i}" for disaster in ['eq', 'dr', 'fl', 'st', 'tot']
                        for i in range(12)]
    missing_cols = [col for col in required_columns if col not in df_dis.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    for disaster in ['eq', 'dr', 'fl', 'st']:
        params = dict_dis_par.get(disaster)
        if not params or len(params) != 12:
            raise ValueError(f"Need exactly 12 parameters for {disaster}")
        disaster_cols = [f"{disaster}_{i}" for i in range(12)]
        weights = np.array([params[i] for i in range(12)])
        impacts = df_dis[disaster_cols].values.dot(weights)
        df_dis[f"{disaster}_score"] = impacts
    return df_dis
def parse_age_group(age_group_str):
      """Helper function to parse age_group.
         This expects strings like "20-24". """
      lower, upper = map(int, age_group_str.split('-'))
      return lower, upper

def simulate_row_grouped_deterministic(row, separate_disasters = False, group_size=25):
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
        theta = (param_nta * nta_values) + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['eq_score']) + (row['fl_score']) + (row['st_score']) + (row['dr_score'])
    else:
        theta = (param_nta * nta_values) + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['tot_score'])

    # Compute remittance probability using the logistic transformation.
    p = 1 / (1 + np.exp(-theta))
    # p[nta_values == 0] = 0 # Set probability to zero if nta is zero.

    # Simulate the remittance decision (1: sends remittance, 0: does not).
    total_senders = int(sum(p))

    # Calculate the total remitted amount for this row.
    total_remittance = total_senders * fixed_remittance * group_size
    return total_remittance

#########################3
values_a = np.linspace(-1, 1, 11)
values_c = np.linspace(0, 1.5, 7)

def compute_disasters_scores_all_countries(df, values_a, values_c):
    df_list = []
    for disaster in ['tot']:
        for a in values_a:
            for c in values_c:
                df_disaster = pd.DataFrame([])
                params = [sin_function_simple(a, c, x) for x in np.linspace(0, 11, 12)]
                disaster_cols = [f"{disaster}_{i}" for i in range(12)]
                weights = np.array([params[i] for i in range(12)])
                impacts = df[disaster_cols].values.dot(weights)
                df_disaster['origin'] = df['origin']
                df_disaster['date'] = df['date']
                df_disaster["value_a"] = round(a,2)
                df_disaster["value_c"] = round(c,2)
                df_disaster["disaster"] = disaster
                df_disaster[f"{disaster}_score"] = impacts
                df_list.append(df_disaster)
    df_output = pd.concat(df_list)
    return df_output

out = compute_disasters_scores_all_countries(emdat, values_a, values_c)
out.to_pickle("C:\\Data\\my_datasets\\disaster_scores_only_tot.pkl")

df_scores = pd.read_pickle("C:\\Data\\my_datasets\\disaster_scores_only_tot.pkl")

##################################
# parameter space
param_nta_space = np.linspace(0.5, 1.5, 6)
param_stay_space = np.linspace(-0.8, 0, 6)
param_asy_space = np.linspace(-4, -2, 6)
param_gdp_space = np.linspace(0, 1, 6)
fixed_remittance_space = [900,1100, 1300]  # Amount each sender sends

###########################
# check initial guess
origin_countries = (df[df.destination == "Italy"]['origin'].unique().tolist())
origin_countries.remove("Cote d'Ivoire")
countries_high_remittances = df_rem_group[df_rem_group.remittances > 1_000_000].origin.unique().tolist()
all_countries = list(set(origin_countries).intersection(set(countries_high_remittances)))

param_nta = 1
param_stay = -0.2
param_asy = -3.5
param_gdp = 0.5
fixed_remittance = 1100  # Amount each sender sends
a = 0.6
c = 1.

def plot_country_mean(df):
    df_mean = df[['origin', 'remittances', 'sim_remittances']].groupby(['origin']).mean().reset_index()
    fig = px.scatter(df_mean, x = 'remittances', y = 'sim_remittances',
                     color = 'origin', log_x=True, log_y=True)
    fig.add_scatter(x=np.linspace(0, df_mean.remittances.max(), 100),
                    y=np.linspace(0, df_mean.remittances.max(), 100))
    fig.show()
    goodness_of_fit_results(df_mean)

def check_initial_guess():
    countries = all_countries
    global nta_dict
    # df country
    df_country = df.query(f"""`origin` in {countries} and `destination` == '{destination}'""")
    df_country = df_country[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    # asy
    asy_df_country = asy_df.query(f"""`destination` == '{destination}'""")
    df_country = df_country.merge(asy_df_country[["date", "asymmetry", "origin"]],
                                  on=["date", "origin"], how='left').ffill()
    # growth rates
    growth_rates_cr = growth_rates.query(f"""`destination` == '{destination}'""")
    df_country = df_country.merge(growth_rates_cr[["date", "yrly_growth_rate", "origin"]],
                                  on=["date", "origin"], how='left')
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].bfill()
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country = df_country.merge(df_betas, on="yrly_growth_rate", how='left')
    ##gdp diff
    df_gdp_cr = df_gdp.query(f"""`destination` == '{destination}'""")
    df_country = df_country.merge(df_gdp_cr[["date", "gdp_diff_norm", "origin"]], on=["date", "origin"],
                                  how='left')
    df_country['gdp_diff_norm'] = df_country['gdp_diff_norm'].bfill()
    ## nta
    df_nta_country = df_nta.query(f"""`country` == '{destination}'""")[['age', 'nta']].fillna(0)
    emdat_ = df_scores[df_scores.origin.isin(countries)]

    for ind, row in df_nta_country.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)
    dis_params['tot'] = [a, c]
    try:
        df_country.drop(columns=f"tot_score", inplace=True)
    except:
        pass
    df_country = df_country.merge(
        emdat_[(emdat_.value_a == a) & (emdat_.value_c == c)]
        [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
    df_country['tot_score'] = df_country['tot_score'].fillna(0)

    df_country['sim_remittances'] = df_country.apply(simulate_row_grouped_deterministic, axis=1)
    remittance_per_period = df_country.groupby(['date', 'origin'])['sim_remittances'].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin'], how="left")
    remittance_per_period['error_squared'] = (remittance_per_period['remittances'] -
                                              remittance_per_period['sim_remittances']) ** 2
    goodness_of_fit_results(remittance_per_period)

    plot_country_mean(remittance_per_period)


check_initial_guess()
##########################
values_a = [0]# np.linspace(-1, 1, 2)
values_c = [0]# np.linspace(0, 1.5, 2)
param_nta_space = [round(x,2) for x in np.linspace(0.5, 1.5, 4)]
param_stay_space = [round(x,2) for x in np.linspace(-0.8, 0, 4)]
param_asy_space = [round(x,2) for x in np.linspace(-4, -2, 4)]
param_gdp_space = [round(x,2) for x in np.linspace(0, 1, 4)]
fixed_remittance_space = [800]  # Amount each sender sends

#############
results_list = []
n_repetitions = 10
for f in tqdm(range(n_repetitions)):
    countries = sample(all_countries, int(len(all_countries) * 0.6))
    global nta_dict
    # df country
    df_country = df.query(f"""`origin` in {countries} and `destination` == '{destination}'""")
    df_country = df_country[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    # asy
    asy_df_country = asy_df.query(f"""`destination` == '{destination}'""")
    df_country = df_country.merge(asy_df_country[["date", "asymmetry", "origin"]],
                                  on=["date", "origin"], how='left').ffill()
    # growth rates
    growth_rates_cr = growth_rates.query(f"""`destination` == '{destination}'""")
    df_country = df_country.merge(growth_rates_cr[["date", "yrly_growth_rate", "origin"]],
                                  on=["date", "origin"], how='left')
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].bfill()
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country = df_country.merge(df_betas, on="yrly_growth_rate", how='left')
    ##gdp diff
    df_gdp_cr = df_gdp.query(f"""`destination` == '{destination}'""")
    df_country = df_country.merge(df_gdp_cr[["date", "gdp_diff_norm", "origin"]], on=["date", "origin"],
                                  how='left')
    df_country['gdp_diff_norm'] = df_country['gdp_diff_norm'].bfill()
    ## nta
    df_nta_country = df_nta.query(f"""`country` == '{destination}'""")[['age', 'nta']].fillna(0)
    for ind, row in df_nta_country.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)
    emdat_ = df_scores[df_scores.origin.isin(countries)]
    df_country_ = df_country.copy()

    for param_nta in tqdm(param_nta_space):
        for param_asy in param_asy_space:
            for param_stay in param_stay_space:
                for param_gdp in param_gdp_space:
                    for fixed_remittance in fixed_remittance_space:
                        for a in values_a:
                            for c in values_c:
                                dis_params['tot'] = [a,c]
                                try:
                                    df_country.drop(columns=f"tot_score", inplace=True)
                                except:
                                    pass
                                df_country = df_country.merge(
                                    emdat_[(emdat_.disaster == 'tot') & (emdat_.value_a == a) & (emdat_.value_c == c)]
                                    [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
                                df_country['tot_score'] = df_country['tot_score'].fillna(0)

                                df_country['sim_remittances'] = df_country.apply(simulate_row_grouped_deterministic, axis=1)
                                remittance_per_period = df_country.groupby(['date', 'origin'])['sim_remittances'].sum().reset_index()
                                remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin'], how="left")
                                remittance_per_period['error_squared'] = (remittance_per_period['remittances'] -
                                                                          remittance_per_period['sim_remittances']) ** 2

                                dict_params = {"nta" : param_nta,
                                               "asy" : param_asy,
                                               "stay" : param_stay,
                                               "gdp" : param_gdp,
                                               "a" : a,
                                               "c" : c,
                                               "rem_value" : fixed_remittance}
                                results_run = [dict_params, remittance_per_period['error_squared'].mean()] + [f]
                                results_list.append(results_run)

                                df_country = df_country_.copy()

import pickle
with open('model_results.pkl', 'wb') as fi:
    pickle.dump(results_list, fi)

with open('model_results.pkl', 'rb') as fi:
    loaded_data = pickle.load(fi)
################

min_tuple_each_run = []
for f in tqdm(range(n_repetitions)):
    sub_data = [x for x in results_list if x[2] == f]
    flattened_data = [item[1] for item in sub_data]
    min_tuple_index = flattened_data.index(min(flattened_data))
    min_tuple = sub_data[min_tuple_index]
    min_tuple_each_run.append(min_tuple)

def r_squared_all_and_mean():
    countries = all_countries
    global nta_dict
    # df country
    df_country = df.query(f"""`origin` in {countries} and `destination` == '{destination}'""")
    df_country = df_country[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    # asy
    asy_df_country = asy_df.query(f"""`destination` == '{destination}'""")
    df_country = df_country.merge(asy_df_country[["date", "asymmetry", "origin"]],
                                  on=["date", "origin"], how='left').ffill()
    # growth rates
    growth_rates_cr = growth_rates.query(f"""`destination` == '{destination}'""")
    df_country = df_country.merge(growth_rates_cr[["date", "yrly_growth_rate", "origin"]],
                                  on=["date", "origin"], how='left')
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].bfill()
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country = df_country.merge(df_betas, on="yrly_growth_rate", how='left')
    ##gdp diff
    df_gdp_cr = df_gdp.query(f"""`destination` == '{destination}'""")
    df_country = df_country.merge(df_gdp_cr[["date", "gdp_diff_norm", "origin"]], on=["date", "origin"],
                                  how='left')
    df_country['gdp_diff_norm'] = df_country['gdp_diff_norm'].bfill()
    ## nta
    df_nta_country = df_nta.query(f"""`country` == '{destination}'""")[['age', 'nta']].fillna(0)
    emdat_ = df_scores[df_scores.origin.isin(countries)]

    for ind, row in df_nta_country.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)
    dis_params['tot'] = [a, c]
    try:
        df_country.drop(columns=f"tot_score", inplace=True)
    except:
        pass
    df_country = df_country.merge(
        emdat_[(emdat_.value_a == a) & (emdat_.value_c == c)]
        [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
    df_country['tot_score'] = df_country['tot_score'].fillna(0)

    df_country['sim_remittances'] = df_country.apply(simulate_row_grouped_deterministic, axis=1)
    remittance_per_period = df_country.groupby(['date', 'origin'])['sim_remittances'].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin'], how="left")
    remittance_per_period['error_squared'] = (remittance_per_period['remittances'] -
                                              remittance_per_period['sim_remittances']) ** 2
    remittance_per_period['error'] = remittance_per_period['remittances'] - remittance_per_period['sim_remittances']
    SS_res = np.sum(np.square(remittance_per_period['error']))
    SS_tot = np.sum(np.square(remittance_per_period['remittances'] - np.mean(remittance_per_period['remittances'])))
    R_squared = 1 - (SS_res / SS_tot)
    print(f"R-squared: {round(R_squared, 3)}")
    df_mean = remittance_per_period[['origin', 'remittances', 'sim_remittances']].groupby(['origin']).mean().reset_index()
    df_mean['error'] = df_mean['remittances'] - df_mean['sim_remittances']
    SS_res = np.sum(np.square(df_mean['error']))
    SS_tot = np.sum(np.square(df_mean['remittances'] - np.mean(df_mean['remittances'])))
    R_squared = 1 - (SS_res / SS_tot)
    print(f"R-squared means: {round(R_squared, 3)}")

for min_tuple in min_tuple_each_run:
    best_params = min_tuple[0]
    print(best_params)
    print(f"Abs error: {min_tuple[1] / 1_000_000_000_000}")
    param_nta = best_params["nta"]
    param_stay = best_params["stay"]
    param_asy = best_params["asy"]
    param_gdp = best_params["gdp"]
    a = best_params["a"]
    c = best_params["c"]
    fixed_remittance = best_params["rem_value"]
    r_squared_all_and_mean()

check_initial_guess()