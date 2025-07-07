
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time
from random import sample

## pair of countries
origin, destination = "Philippines", "Japan"

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_interpolated.pkl"
df = pd.read_pickle(diasporas_file)

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
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters.pkl").rename(columns = {'country' : 'origin'})

#########################################
#########################################
# Sample parameters
param_nta = 0.8
param_stay = -0.1
param_asy = -2
param_gdp = 8
fixed_remittance = 1100  # Amount each sender sends
##### disasters parameters

dis_params = pd.read_excel("C:\\Data\\my_datasets\\disasters\\disasters_params.xlsx", sheet_name="Sheet2").dropna()

def sin_function(a,b,c,x):
    return a * np.sin((np.pi/6) * x) + b * np.sin((np.pi/3) * x) + c

def sin_function_simple(a,c,x):
    return a + np.sin((np.pi/c) * x)

def zero_after_second_zero(lst):
    zero_count = 0
    for i in range(len(lst)):
        if lst[i] == 0:
            zero_count += 1
            if zero_count == 2:
                # Set all remaining elements to 0
                lst[i+1:] = [0] * (len(lst) - i - 1)
                break
    return lst

def disaster_score_function(disasters = ['eq', 'dr', 'fl', 'st'], simple = True):
    global dict_dis_par
    dict_dis_par = {}
    for dis in disasters:
        if not simple:
            a,b,c = dis_params[dis]
            values = [sin_function(a,b,c,x) for x in np.linspace(0, 11, 12)]
            values = zero_after_second_zero(values.copy())
        else:
            a,c = dis_params[dis]
            values = [sin_function_simple(a,c,x) for x in np.linspace(0, 11, 12)]
            values = zero_after_second_zero(values.copy())
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
      p[nta_values == 0] = 0 # Set probability to zero if nta is zero.

      # Simulate the remittance decision (1: sends remittance, 0: does not).
      decisions = np.random.binomial(1, p)
      total_senders = sum(decisions)

      # Calculate the total remitted amount for this row.
      total_remittance = total_senders * fixed_remittance * group_size
      return total_remittance

def probabilities_row_grouped(row, group_size = 25):
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
    p[nta_values == 0] = 0
    return p

def simulate_country_pair(origin, destination, df, disasters = True):
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
    return df_country, remittance_per_period

def plot_remittances_pair(origin, destination, disasters = True, yearly = False):
      results, remittance_per_period = simulate_country_pair(origin, destination, df, disasters = disasters)

      if yearly:
          remittance_per_period = remittance_per_period.groupby(remittance_per_period.date.dt.year)['sim_remittances'].sum().reset_index()
          remittance_per_period = remittance_per_period[remittance_per_period.date != 2020]
      # Plot
      plt.figure(figsize=(12, 6))
      plt.plot(remittance_per_period['date'], remittance_per_period['sim_remittances'], marker='o')
      plt.title(f'Total Remittances per Period, from {destination} to {origin}')
      plt.xlabel('Date')
      plt.ylabel('Total Remittances')
      plt.grid(True)
      plt.tight_layout()
      plt.show(block = True)


#########################################
#########################################
# biggest diaspora groups
pair_count = (df[['origin', 'destination', 'n_people']].groupby(['origin', 'destination']).sum().
              reset_index().sort_values('n_people', ascending = False))

def return_population_profile(origin, destination, disasters = True):
    global nta_dict
    # df country
    df_country = df.query(f"""`origin` == '{origin}' and  `destination` == '{destination}'""")
    df_country = df_country[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
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
        df_country = df_country.merge(emdat_country, on='date', how='left').fillna(0)
    else:
        for dis in ['dr', 'eq', 'fl', 'st', 'tot']:
            df_country[dis + "_score"] = 0

    for ind, row in df_nta_country.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)

    #### simulate
    df_country['probabilities'] = df_country.apply(probabilities_row_grouped, axis=1)
    return df_country['probabilities'].explode().tolist()

interesting_groups = [["Mexico", "USA"], ["Russia", "Ukraine"], ["Afghanistan","United Arab Emirates"], ["Bangladesh" , "India"]]
results = {}
for group in tqdm(interesting_groups):
    origin, destination = group
    prob_list = return_population_profile(origin, destination, disasters = True)
    results[origin] = prob_list

plt.figure(figsize=(10, 6))

for label, probs in results.items():
    sampled_probs = sample([x for x in probs if x != np.nan],8_000)
    sampled_probs = np.sort(sampled_probs)
    x = np.linspace(0, 1, len(sampled_probs))
    plt.plot(x, sampled_probs, label=label, linewidth=3)

plt.title("Normalized Remittance Probability by Individual")
plt.xlabel("Normalized Individual Position (0 = first, 1 = last)")
plt.ylabel("Probability of Sending Remittances")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('C:\\Users\\Andrea Vismara\\Desktop\\papers\\remittances\\charts\\profiles_with_disasters.pdf', bbox_inches = 'tight')  # Save for PowerPoint
plt.show(block = True)











