

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time
import itertools
from random import sample

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
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags.pkl")


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
    return a + np.sin((np.pi/c) * x)


def zero_values_before_first_positive_and_after_last_negative(lst):
    # Find index of first positive value
    first_positive = next((i for i, x in enumerate(lst) if x > 0), None)

    # Find index of last negative value
    last_negative = len(lst) - next((i for i, x in enumerate(reversed(lst)) if x < 0), len(lst)) - 1

    # Create a copy of the list to modify
    modified = lst.copy()

    # Set values before first positive to zero
    if first_positive is not None:
        for i in range(first_positive):
            modified[i] = 0

    # Set positive values after last negative to zero
    if last_negative >= 0:
        for i in range(last_negative + 1, len(modified)):
            if modified[i] > 0:
                modified[i] = 0

    return modified

def disaster_score_function(disasters = ['eq', 'dr', 'fl', 'st'], simple = True):
    global dict_dis_par
    dict_dis_par = {}
    for dis in disasters:
        if not simple:
            a,b,c = dis_params[dis]
            values = [sin_function(a,b,c,x) for x in np.linspace(0, 11, 12)]
            values = zero_values_before_first_positive_and_after_last_negative(values.copy())
        else:
            a,c = dis_params[dis]
            values = [sin_function_simple(a,c,x) for x in np.linspace(0, 11, 12)]
            values = zero_values_before_first_positive_and_after_last_negative(values.copy())
        dict_dis_par[dis] = values
    return dict_dis_par

dict_dis_par = disaster_score_function(disasters = ['eq', 'dr', 'fl', 'st'], simple=True)


#####################################
#####################################

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

# Simulation for one row, grouping individuals in batches of 25.
def simulate_row_grouped(row, group_size=25):
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
      theta = (param_nta * nta_values) + (param_stay * yrs_stay) \
              + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
              + (row['eq_score']) + (row['fl_score']) + (row['st_score']) + (row['dr_score'])

      # Compute remittance probability using the logistic transformation.
      p = 1 / (1 + np.exp(-theta))
      # p[nta_values == 0] = 0 # Set probability to zero if nta is zero.

      # Simulate the remittance decision (1: sends remittance, 0: does not).
      decisions = np.random.binomial(1, p)
      total_senders = sum(decisions)

      # Calculate the total remitted amount for this row.
      total_remittance = total_senders * fixed_remittance * group_size
      return total_remittance

def simulate_row_grouped_deterministic(row, group_size=25):
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
    theta = (param_nta * nta_values) + (param_stay * yrs_stay) \
            + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
            + (row['eq_score']) + (row['fl_score']) + (row['st_score']) + (row['dr_score'])

    # Compute remittance probability using the logistic transformation.
    p = 1 / (1 + np.exp(-theta))
    # p[nta_values == 0] = 0 # Set probability to zero if nta is zero.

    # Simulate the remittance decision (1: sends remittance, 0: does not).
    total_senders = int(sum(p))

    # Calculate the total remitted amount for this row.
    total_remittance = total_senders * fixed_remittance * group_size
    return total_remittance

def simulate_country_pair_comparison(origin, destination, df, disasters = True):
    global nta_dict
    # df country
    df_country = df.query(f"""`origin` == '{origin}' and  `destination` == '{destination}'""")
    df_country = df_country[[x for x in df.columns if x != 'sex']].groupby(['date', 'origin', 'age_group', 'mean_age','destination']).mean().reset_index()
    # asy
    asy_df_country = asy_df.query(f"""`origin` == '{origin}' and  `destination` == '{destination}'""")
    df_country = df_country.merge(asy_df_country[["date", "asymmetry"]], on="date", how='left').ffill()
    # growth rates
    growth_rates_cr = growth_rates.query(f"""`origin` == '{origin}' and  `destination` == '{destination}'""")
    df_country = df_country.merge(growth_rates_cr[["date", "yrly_growth_rate"]], on="date", how='left')
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].bfill()
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country = df_country.merge(df_betas, on="yrly_growth_rate", how='left')
    ##gdp diff
    df_gdp_cr = df_gdp.query(f"""`origin` == '{origin}' and  `destination` == '{destination}'""")
    df_country = df_country.merge(df_gdp_cr[["date", "gdp_diff_norm"]], on="date", how='left')
    df_country['gdp_diff_norm'] = df_country['gdp_diff_norm'].bfill()
    ## nta
    df_nta_country = df_nta.query(f"""`country` == '{destination}'""")[['age', 'nta']].fillna(0)
    ## disasters
    if disasters:
        emdat_country = emdat.query(f"""`origin` == '{origin}'""")
        df_country_dis = df_country[~df_country['date'].duplicated()][['date', 'origin']]
        emdat_country = df_country_dis.merge(emdat_country, on=['origin', 'date'], how='left').fillna(0)
        emdat_country = compute_disasters_scores(emdat_country, dict_dis_par)
        df_country = df_country.merge(emdat_country, on = 'date', how = 'left').fillna(0)
    else:
        for dis in ['dr', 'eq', 'fl', 'st' , 'tot']:
            df_country[dis+"_score"] = 0

    for ind, row in df_nta_country.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)

    #### simulate
    df_country['sim_remittances'] = df_country.apply(simulate_row_grouped, axis=1)
    remittance_per_period = df_country.groupby('date')['sim_remittances'].sum().reset_index()
    remittance_per_period['origin'] = origin
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin'], how="left")
    return df_country, remittance_per_period

def simulate_all_countries_comparison(countries, destination, df, disasters = True):
    global nta_dict
    # df country
    df_country = df.query(f"""`origin` in {countries} and `destination` == '{destination}'""")
    df_country = df_country[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    # asy
    asy_df_country = asy_df.query(f"""`destination` == '{destination}'""")
    df_country = df_country.merge(asy_df_country[["date", "asymmetry", "origin"]], on=["date", "origin"], how='left').ffill()
    # growth rates
    growth_rates_cr = growth_rates.query(f"""`destination` == '{destination}'""")
    df_country = df_country.merge(growth_rates_cr[["date", "yrly_growth_rate", "origin"]], on=["date", "origin"], how='left')
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].bfill()
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country = df_country.merge(df_betas, on="yrly_growth_rate", how='left')
    ##gdp diff
    df_gdp_cr = df_gdp.query(f"""`destination` == '{destination}'""")
    df_country = df_country.merge(df_gdp_cr[["date", "gdp_diff_norm", "origin"]], on=["date", "origin"], how='left')
    df_country['gdp_diff_norm'] = df_country['gdp_diff_norm'].bfill()
    ## nta
    df_nta_country = df_nta.query(f"""`country` == '{destination}'""")[['age', 'nta']].fillna(0)
    ## disasters
    if disasters:
        emdat_country = emdat.copy()
        df_country_dis = df_country[~df_country[['date', 'origin']].duplicated()][['date', 'origin']]
        emdat_country = df_country_dis.merge(emdat_country, on=['origin', 'date'], how='left').fillna(0).sort_values(['origin', 'date'])
        emdat_country = compute_disasters_scores_all_countries(emdat_country, dict_dis_par)
        df_country = df_country.merge(emdat_country, on=['origin', 'date'], how='left').fillna(0)
    else:
        for dis in ['dr', 'eq', 'fl', 'st', 'tot']:
            df_country[dis + "_score"] = 0

    for ind, row in df_nta_country.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)

    #### simulate
    df_country['sim_remittances'] = df_country.apply(simulate_row_grouped_deterministic, axis=1)
    remittance_per_period = df_country.groupby(['date', 'origin'])['sim_remittances'].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin'], how="left")
    return remittance_per_period

def plot_remittances_pair_comparison(origin, destination, disasters = True, yearly = False):
      results, remittance_per_period = simulate_country_pair_comparison(origin, destination, df, disasters = disasters)

      if yearly:
          remittance_per_period = remittance_per_period.groupby(remittance_per_period.date.dt.year)['sim_remittances'].sum().reset_index()
          remittance_per_period = remittance_per_period[remittance_per_period.date != 2020]
      # Plot
      plt.figure(figsize=(12, 6))
      plt.plot(remittance_per_period['date'], remittance_per_period['sim_remittances'], marker='o', label = "simulated")
      plt.plot(remittance_per_period['date'], remittance_per_period['remittances'], marker='o', label = 'observed')
      plt.title(f'Total Remittances per Period, from {destination} to {origin}')
      plt.xlabel('Date')
      plt.ylabel('Remittances')
      plt.grid(True)
      plt.tight_layout()
      plt.show(block = True)

def plot_all_countries_comparison(countries, destination, disasters = True):
    remittance_per_period = simulate_all_countries_comparison(countries, destination, df, disasters=True)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(remittance_per_period['remittances'], remittance_per_period['sim_remittances'], alpha=0.6)
    plt.xlabel('Observed Remittances')
    plt.ylabel('Simulated Remittances')
    plt.title("Comparison of results")
    # Add identity line
    lims = [0, remittance_per_period['remittances'].max()]
    ax.plot(lims, lims, 'k-', alpha=1, zorder=1)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.show(block = True)

plot_remittances_pair_comparison(origin = "Philippines", destination = "Italy", yearly = False)

########################
# iterate over parameters space or disasters
########################

values = np.linspace(0, 0.2, 3)
param_names = ['eq', 'dr', 'fl', 'st']
combinations = list(itertools.permutations(values))

origin_countries = (df[df.destination == "Italy"]['origin'].unique().tolist())
origin_countries.remove("Cote d'Ivoire")
countries_high_remittances = df_rem_group[df_rem_group.remittances > 100_000].origin.unique().tolist()
countries = list(set(origin_countries).intersection(set(countries_high_remittances)))

def run_model(countries, destination):
    remittance_per_period = simulate_all_countries_comparison(countries, destination, df, disasters = True)
    remittance_per_period['error_squared'] = (remittance_per_period['remittances'] - remittance_per_period['sim_remittances'])**2
    return dis_params.iloc[:,:4].to_numpy(), remittance_per_period['error_squared'].mean()


####### run for Italy

values_a = [-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
values_c = [4, 6, 8, 10]

def compute_disasters_scores_all_countries(df, values_a, values_c):
    df_list = []
    for disaster in ['eq', 'dr', 'fl', 'st']:
        for a in values_a:
            for c in values_c:
                df_disaster = pd.DataFrame([])
                params = [sin_function_simple(a, c, x) for x in np.linspace(0, 11, 12)]
                disaster_cols = [f"{disaster}_{i}" for i in range(12)]
                weights = np.array([params[i] for i in range(12)])
                impacts = df[disaster_cols].values.dot(weights)
                df_disaster['origin'] = df['origin']
                df_disaster['date'] = df['date']
                df_disaster["value_a"] = a
                df_disaster["value_c"] = c
                df_disaster["disaster"] = disaster
                df_disaster[f"{disaster}_score"] = impacts
                df_list.append(df_disaster)
    df_output = pd.concat(df_list)
    return df_output

out = compute_disasters_scores_all_countries(emdat, values_a, values_c)
out.to_pickle("C:\\Data\\my_datasets\\disaster_scores.pkl")

df_scores = pd.read_pickle("C:\\Data\\my_datasets\\disaster_scores.pkl")

param_names = ['eq', 'dr', 'fl', 'st']

origin_countries = (df[df.destination == "Italy"]['origin'].unique().tolist())
origin_countries.remove("Cote d'Ivoire")
countries_high_remittances = df_rem_group[df_rem_group.remittances > 100_000].origin.unique().tolist()
all_countries = list(set(origin_countries).intersection(set(countries_high_remittances)))

results_list = []

for f in tqdm(range(6)):
    countries = sample(all_countries, 45)
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
    for disaster in tqdm(['eq', 'dr', 'fl', 'st']):
        df_country = df_country.merge(
            emdat_[(emdat_.disaster == disaster) & (emdat_.value_a == 0.1) & (emdat_.value_c == 4)]
            [[f"{disaster}_score", "origin", "date"]], on=["date", "origin"], how="left")
        dis_params[disaster] = [0, 2]
    df_country_ = df_country.copy()

    for disaster in tqdm(['eq', 'dr', 'fl', 'st']):
        for a in values_a:
            for c in values_c:
                dis_params[disaster] = [a,c]
                df_country.drop(columns = f"{disaster}_score", inplace = True)
                df_country = df_country.merge(
                    emdat_[(emdat_.disaster == disaster) & (emdat_.value_a == a) & (emdat_.value_c == c)]
                    [[f"{disaster}_score", "origin", "date"]], on=["date", "origin"], how="left")

                df_country['sim_remittances'] = df_country.apply(simulate_row_grouped_deterministic, axis=1)
                remittance_per_period = df_country.groupby(['date', 'origin'])['sim_remittances'].sum().reset_index()
                remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin'], how="left")
                remittance_per_period['error_squared'] = (remittance_per_period['remittances'] -
                                                          remittance_per_period['sim_remittances']) ** 2

                results_run = [dis_params.iloc[:,:4].to_numpy(), remittance_per_period['error_squared'].mean()] + [f]
                results_list.append(results_run)

                df_country = df_country_.copy()

import pickle
with open('model_results.pkl', 'wb') as fi:
    pickle.dump(results_list, fi)

with open('model_results.pkl', 'rb') as fi:
    loaded_data = pickle.load(fi)

# Flatten the list (since each element is a single-item list containing a tuple)
min_tuple_each_run = []
for f in tqdm(range(6)):
    sub_data = [x for x in loaded_data if x[2] == f]
    flattened_data = [item[1] for item in sub_data]
    min_tuple_index = flattened_data.index(min(flattened_data))
    min_tuple = sub_data[min_tuple_index]
    min_tuple_each_run.append(min_tuple)


for i in range(len(min_tuple[0])):
    dis_params.iloc[i, :4] = min_tuple[0][i]

#########
# plot disasters impacts
#############
from scipy.stats import sem, t

disaster_dicts = []
for min_tuple in min_tuple_each_run:
    for i in range(len(min_tuple[0])):
        dis_params.iloc[i, :4] = min_tuple[0][i]

    dict_dis_par = disaster_score_function(disasters=['eq', 'dr', 'fl', 'st'])
    disaster_dicts.append(dict_dis_par)

disasters = disaster_dicts[0].keys()
results = {}

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()  # Flatten to 1D array for easy iteration

# Plot each category in a subplot
for i, (category, values) in enumerate(results.items()):
    ax = axes[i]
    x = np.arange(len(values['mean']))  # X-axis as indices
    mean = values['mean']
    ci_lower = values['ci_lower']
    ci_upper = values['ci_upper']

    # Plot mean and confidence interval
    ax.plot(x, mean, label='Mean', color='blue', marker='o')
    ax.fill_between(x, ci_lower, ci_upper, alpha=0.3, color='blue', label='95% CI')

    # Customize subplot
    ax.set_title(category.upper())
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show(block = True)


#
# df_dis = pd.DataFrame(dict_dis_par)
# periods = range(0, 12)
#
# # Set up the plot
# plt.figure(figsize=(10, 6))
#
# # Plot each category
# for column in df_dis.columns:
#     plt.plot(periods, df_dis[column], marker='o', label=column)
#
# # Customize the plot
# plt.title("Disaster Impacts Over 12 Periods", fontsize=14)
# plt.xlabel("Period", fontsize=12)
# plt.ylabel("Impact Value", fontsize=12)
# plt.xticks(periods)  # Show all periods on x-axis
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(title="Disasters categories")  # Legend outside plot
# plt.savefig('C:\\Users\\Andrea Vismara\\Desktop\\papers\\remittances\\charts\\disasters_effects.pdf', bbox_inches = 'tight')  # Save for PowerPoint
#
# # Show the plot
# plt.show(block = True)