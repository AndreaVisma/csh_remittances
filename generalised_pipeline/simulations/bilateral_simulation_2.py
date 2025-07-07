



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
param_asy = -3.5
param_gdp = 0.5
dis_params = pd.read_excel("C:\\Data\\my_datasets\\disasters\\disasters_params.xlsx")
dict_dis_par = dict(zip(['eq', 'dr', 'fl', 'st', 'tot'], [dis_params[x].tolist() for x in ['eq', 'dr', 'fl', 'st', 'tot']]))

fixed_remittance = 400  # Amount each sender sends

#####################################
#####################################

def compute_disasters_scores(df, dict_dis_par):
    df_dis = df.copy()
    df_dis = df_dis.drop_duplicates(subset=["date", "origin"])
    for col in ['eq', 'dr', 'fl', 'st']:
        for shift in [int(x) for x in np.linspace(1, 12, 12)]:
            g = df_dis.groupby('origin', group_keys=False)
            g = g.apply(lambda x: x.set_index(['date', 'origin'])[col]
                        .shift(shift).reset_index(drop=True)).fillna(0)
            df_dis[f'{col}_{shift}'] = g.iloc[0]
            df_dis['tot'] = df_dis['fl'] + df_dis['eq'] + df_dis['st'] + df_dis['dr']
    for shift in [int(x) for x in np.linspace(1, 12, 12)]:
        df_dis[f'tot_{shift}'] = df_dis[f'fl_{shift}'] + df_dis[f'eq_{shift}'] + df_dis[f'st_{shift}'] + df_dis[
            f'dr_{shift}']
    df_dis.rename(columns={'eq': 'eq_0', 'st': 'st_0', 'fl': 'fl_0', 'dr': 'dr_0', 'tot': 'tot_0'}, inplace=True)
    required_columns = ['date', 'origin'] + \
                       [f"{disaster}_{i}" for disaster in ['eq', 'dr', 'fl', 'st', 'tot']
                        for i in range(13)]
    missing_cols = [col for col in required_columns if col not in df_dis.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    for disaster in ['eq', 'dr', 'fl', 'st', 'tot']:
        params = dict_dis_par.get(disaster)
        if not params or len(params) != 13:
            raise ValueError(f"Need exactly 13 parameters for {disaster}")
        disaster_cols = [f"{disaster}_{i}" for i in range(13)]
        weights = np.array([params[i] for i in range(13)])
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

plot_remittances_pair(origin = "Philippines", destination = "Italy", yearly = False)


#############################################
#############################################
#### USA TO MEXICO
results_with, rem_per_period_with = simulate_country_pair("Mexico", "USA", df, disasters=True)
results_no, rem_per_period_no = simulate_country_pair("Mexico", "USA", df, disasters=False)

n_people_df = df.query(f"""`origin` == 'Mexico' and  `destination` == 'USA'""")
n_people_df = n_people_df[["date", "n_people"]].groupby('date').sum().reset_index()
n_people_df.n_people.plot()
plt.show(block = True)

##load the data in
df_rem = pd.read_excel("c:\\data\\remittances\\mexico\\mexico_remittances_2024.xlsx",
                   skiprows=9)
df_rem = df_rem.iloc[8:,:]
names_column = ["date",
                "total_mln", "money_orders_mln", "checks_mln", "electronic_transfers_mln", "cash_goods_mln",
                "total_operations", "money_orders_operations", "checks_operations", "electronic_transfers_operations", "cash_goods_operations",
                "total_mean_op", "money_orders_mean_op", "checks_mean_op", "electronic_transfers_mean_op", "cash_goods_mean_op"]
dict_columns = dict(zip(df_rem.columns,names_column))
df_rem.rename(columns = dict_columns, inplace = True)
df_rem['date'] = pd.to_datetime(df_rem.date)
df_rem = df_rem[df_rem.date.dt.year >= 2010]
df_rem = df_rem[df_rem.date.dt.year <= 2019]

# Create the main plot and secondary axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary y-axis: simulated remittances
ax1.plot(rem_per_period_no['date'], rem_per_period_no['sim_remittances'], marker='o', label="Simulation without disasters", color='tab:blue')
ax1.plot(rem_per_period_with['date'], rem_per_period_with['sim_remittances'], marker='o', label="Simulation with disasters", color='tab:orange')
ax1.set_xlabel('Date')
ax1.set_ylabel('Simulated Remittances', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True)

# Secondary y-axis: actual remittance totals from df_rem
ax2 = ax1.twinx()
ax2.plot(df_rem['date'], df_rem['total_mln'], color='tab:green', linestyle='--', label='Reported total remittances (millions)')
ax2.set_ylabel('Reported Remittances (Million USD)', color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')

# Title and legend
fig.suptitle('Total Remittances per Period, from USA to Mexico')
fig.legend(loc = 'upper left', bbox_to_anchor=(0.15, 0.85))  # Place legend manually for clarity
plt.show(block = True)

###########################################
###########################################
## Effect of individual disasters types

destination = "USA"
origin = "Mexico"
dis_params = pd.read_excel("C:\\Data\\my_datasets\\disasters\\disasters_params.xlsx")
for col in ['dr', 'fl', 'st']:
    dis_params[col] = 0
dict_dis_par = dict(zip(['eq', 'dr', 'fl', 'st', 'tot'], [dis_params[x].tolist() for x in ['eq', 'dr', 'fl', 'st', 'tot']]))
results_eq, rem_per_period_eq = simulate_country_pair("Mexico", "USA", df, disasters=True)
dis_params = pd.read_excel("C:\\Data\\my_datasets\\disasters\\disasters_params.xlsx")
for col in ['eq', 'fl', 'st']:
    dis_params[col] = 0
dict_dis_par = dict(zip(['eq', 'dr', 'fl', 'st', 'tot'], [dis_params[x].tolist() for x in ['eq', 'dr', 'fl', 'st', 'tot']]))
results_dr, rem_per_period_dr = simulate_country_pair("Mexico", "USA", df, disasters=True)
dis_params = pd.read_excel("C:\\Data\\my_datasets\\disasters\\disasters_params.xlsx")
for col in ['dr', 'eq', 'st']:
    dis_params[col] = 0
dict_dis_par = dict(zip(['eq', 'dr', 'fl', 'st', 'tot'], [dis_params[x].tolist() for x in ['eq', 'dr', 'fl', 'st', 'tot']]))
results_fl, rem_per_period_fl = simulate_country_pair("Mexico", "USA", df, disasters=True)
dis_params = pd.read_excel("C:\\Data\\my_datasets\\disasters\\disasters_params.xlsx")
for col in ['dr', 'fl', 'eq']:
    dis_params[col] = 0
dict_dis_par = dict(zip(['eq', 'dr', 'fl', 'st', 'tot'], [dis_params[x].tolist() for x in ['eq', 'dr', 'fl', 'st', 'tot']]))
results_st, rem_per_period_st = simulate_country_pair("Mexico", "USA", df, disasters=True)

sum_eq, sum_dr, sum_fl, sum_st = rem_per_period_eq.sim_remittances.sum(), rem_per_period_dr.sim_remittances.sum(), rem_per_period_fl.sim_remittances.sum(), rem_per_period_st.sim_remittances.sum()
sum_no = rem_per_period_no.sim_remittances.sum()
sum_all = rem_per_period_with.sim_remittances.sum()

print(f"Additional remittances in response to earthquakes: {round(100 * (sum_eq - sum_no) / sum_no, 2)}%")
print(f"Additional remittances in response to droughts: {round(100 * (sum_dr - sum_no) / sum_no, 2)}%")
print(f"Additional remittances in response to floods: {round(100 * (sum_fl - sum_no) / sum_no, 2)}%")
print(f"Additional remittances in response to storms: {round(100 * (sum_st - sum_no) / sum_no, 2)}%")
print(f"Additional remittances in response to storms: {round(100 * (sum_all - sum_no) / sum_no, 2)}%")










