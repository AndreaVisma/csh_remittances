



import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time

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
param_nta = 1
param_stay = -0.2
param_asy = -2.5
param_gdp = -2
dis_params = pd.read_excel("C:\\Data\\my_datasets\\disasters\\disasters_params.xlsx")
dict_dis_par = dict(zip(['eq', 'dr', 'fl', 'st', 'tot'], [dis_params[x].tolist() for x in ['eq', 'dr', 'fl', 'st', 'tot']]))

fixed_remittance = 400  # Amount each sender sends

#####################################
#####################################

def compute_disasters_scores(df, dict_dis_par):
    df_dis = df[~df[["date", "origin"]].duplicated()][["date", "origin"] + [x for x in df.columns[2:]]]
    df_dis.rename(columns = {'eq' : 'eq_0', 'st' : 'st_0', 'fl' : 'fl_0', 'dr' : 'dr_0', 'tot' : 'tot_0'}, inplace = True)
    for disaster in ['eq', 'dr', 'fl', 'st', 'tot']:
        params = dict_dis_par[disaster]
        impact =  sum([params[int(x)] * df_dis[f"{disaster}_{int(x)}"] for x in np.linspace(0, 12, 13)])
        df_dis[f"{disaster}_score"] = impact
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
              + (row['tot_score'])

      # Compute remittance probability using the logistic transformation.
      p = 1 / (1 + np.exp(-theta))
      # p[nta_values == 0] = 0 # Set probability to zero if nta is zero.

      # Simulate the remittance decision (1: sends remittance, 0: does not).
      decisions = np.random.binomial(1, p)
      total_senders = sum(decisions)

      # Calculate the total remitted amount for this row.
      total_remittance = total_senders * fixed_remittance * group_size
      return total_remittance

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
        emdat_country = compute_disasters_scores(emdat_country, dict_dis_par)
        df_country = df_country.merge(emdat_country, on = ['date', 'origin'], how = 'left').fillna(0)
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
      # Plot
      plt.figure(figsize=(12, 6))
      plt.plot(remittance_per_period['date'], remittance_per_period['sim_remittances'], marker='o')
      plt.title(f'Total Remittances per Period, from {destination} to {origin}')
      plt.xlabel('Date')
      plt.ylabel('Total Remittances')
      plt.grid(True)
      plt.tight_layout()
      plt.show(block = True)

plot_remittances_pair(origin, destination)

#### JAPAN TO PHILIPPINES
results_with, rem_per_period_with = simulate_country_pair("Mexico", "USA", df, disasters=True)
results_no, rem_per_period_no = simulate_country_pair("Mexico", "USA", df, disasters=False)

plt.figure(figsize=(8, 6))
plt.plot(rem_per_period_no['date'], rem_per_period_no['sim_remittances'], marker='o', label = "Simulation without disasters")
plt.plot(rem_per_period_with['date'], rem_per_period_with['sim_remittances'], marker='o', label = "Simulation with disasters")
plt.title(f'Total Remittances per Period, from USA to Mexico')
plt.xlabel('Date')
plt.ylabel('Total Remittances')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show(block=True)

