

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from pandas.tseries.offsets import MonthEnd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from italy.simulation.func.goodness_of_fit import goodness_of_fit_results
from utils import zero_values_before_first_positive_and_after_first_negative

param_stay = 0

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors.pkl"
df = pd.read_pickle(diasporas_file)
df = df[df.origin != "Libya"]
# df = df.dropna()
df['year'] = df.date.dt.year
obs_count_df = df["date"].value_counts()

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
df.dropna(inplace = True)
df_countries = df[[x for x in df.columns if x != 'sex']].groupby(
    ['date', 'year', 'origin', 'age_group', 'mean_age', 'destination', 'gdp', 'nta']).mean().reset_index()

## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags.pkl")

####parameters

params = [2.725302695640765, -9.769960496731258, 8.310039749519659,
            0.1788188275520765, 0.21924650014050806,-0.7500114294211869,
            0.3856959277016759, 0.1333609093157307]
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

def simulate_remittances(df_countries, height, shape, shift, rem_pct, plot = True, disasters = True):

    if disasters:
        emdat_ita = emdat[emdat.origin.isin(df_countries.origin.unique())].copy()
        emdat_ita = calculate_tot_score(emdat_ita, height, shape, shift)
        try:
            df_countries.drop(columns = 'tot_score', inplace = True)
        except:
            pass
        df_countries = df_countries.merge(emdat_ita, on=['origin', 'date'], how='left')
        df_countries['tot_score'].fillna(0, inplace = True)
    else:
        df_countries['tot_score'] = 0

    df_countries['rem_amount'] = rem_pct * df_countries['gdp'] / 12

    df_countries['sim_senders'] = df_countries.apply(simulate_row_grouped_deterministic, axis=1)
    df_countries['sim_remittances'] = df_countries['sim_senders'] * df_countries['rem_amount']

    remittance_per_period = df_countries.groupby(['date', 'origin', 'destination'])[['sim_remittances', 'sim_senders']].sum().reset_index()

    if plot:
        goodness_of_fit_results(remittance_per_period)

        plot_country_mean(remittance_per_period, two_countries=True)

    return remittance_per_period

##### run simulations
# WITH DISASTERS
df_results = simulate_remittances(df_countries, height, shape, shift, rem_pct, plot = False, disasters = True)
df_results.to_pickle("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_with_disasters.pkl")
# WITHOUT
df_results = simulate_remittances(df_countries, height, shape, shift, rem_pct, plot = False, disasters = False)
df_results.to_pickle("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_without_disasters.pkl")

#### plot

df_with = pd.read_pickle("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_with_disasters.pkl")
df_without = pd.read_pickle("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_without_disasters.pkl")

df_with['quarter'] = df_with.date.dt.to_period('Q').dt.to_timestamp()
df_without['quarter'] = df_without.date.dt.to_period('Q').dt.to_timestamp()

df_with_period = df_with[['quarter', 'sim_remittances']].groupby("quarter").sum().reset_index()
df_with_period['sim_remittances'] /= 1e9
df_with_period.set_index('quarter', inplace = True)
df_without_period = df_without[['quarter', 'sim_remittances']].groupby("quarter").sum().reset_index()
df_without_period['sim_remittances'] /= 1e9
df_without_period.set_index('quarter', inplace = True)

fig, ax = plt.subplots(figsize = (9,6))

plt.plot(df_with_period.iloc[:-1], label = "With disasters")
plt.plot(df_without_period[:-1], label = "Without disasters")
plt.legend()
plt.grid(True)
plt.show(block = True)

## yearly

df_with['year'] = df_with.date.dt.year
df_without['year'] = df_without.date.dt.year

df_with_period = df_with[['year', 'sim_remittances']].groupby("year").sum().reset_index()
df_with_period['sim_remittances'] /= 1e9
df_with_period.set_index('year', inplace = True)
df_without_period = df_without[['year', 'sim_remittances']].groupby("year").sum().reset_index()
df_without_period['sim_remittances'] /= 1e9
df_without_period.set_index('year', inplace = True)

####### world bank data
import os
data_folder = os.getcwd() + "\\data_downloads\\data\\"
#load inflow of remittances
df_in = pd.read_excel(data_folder + "inward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y")
df_in.rename(columns = {"Remittance inflows (US$ million)": "country"}, inplace=True)
df_in = pd.melt(df_in, id_vars=['country'], value_vars=df_in.columns.tolist()[1:])
df_in.rename(columns = {"variable": "year", "value" : "inflow"}, inplace=True)
df_in.year = df_in.year.astype('int')
df_in = df_in[(df_in.year > 2009) & (df_in.year < 2020)]
df_in['inflow'] /= 1_000
df_in = df_in[["year", "inflow"]].groupby('year').sum()

#load outflow of remittances
df_out = pd.read_excel(data_folder + "outward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y", skiprows=2)
df_out.rename(columns = {"Remittance outflows (US$ million)": "country"}, inplace=True)
df_out = pd.melt(df_out, id_vars=['country'], value_vars=df_out.columns.tolist()[1:])
df_out.rename(columns = {"variable": "year", "value" : "outflow"}, inplace=True)
df_out.replace({"2023e": '2023'}, inplace =True)
df_out.year = df_out.year.astype('int')
df_out = df_out[(df_out.year > 2009) & (df_out.year < 2020)]
df_out['outflow'] /= 1_000
df_out = df_out[["year", "outflow"]].groupby('year').sum()

df_wb = df_out.copy()

fig, ax = plt.subplots(figsize = (9,6))

plt.plot(df_with_period.iloc[:-1], label = "Our model with disasters")
plt.plot(df_without_period[:-1], label = "Our model without disasters")
plt.plot(df_in, label = "World Bank inflow estimate")
plt.plot(df_out, label = "World Bank outflow estimate")
plt.ylabel("Total remittances (bn US dollars)")
plt.legend()
plt.grid(True)
plt.show(block = True)

############################
# Analysis
###########################

