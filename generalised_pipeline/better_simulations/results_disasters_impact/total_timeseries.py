

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
from utils import zero_values_before_first_positive_and_after_first_negative, dict_names

param_stay = 0
df_rem_mex = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_renamed.xlsx")[["date", "total_mln"]]
df_rem_mex['date'] = pd.to_datetime(df_rem_mex['date'], format="%Y%m") + MonthEnd(0)
df_rem_mex['origin'] = "Mexico"
df_rem_mex['destination'] = "USA"
df_rem_mex.rename(columns = {'total_mln' : 'remittances'}, inplace = True)
df_rem_mex['remittances'] *= 1_000_000

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors_2207_TRAIN.pkl"
df = pd.read_pickle(diasporas_file)
df = df.dropna()

###gdp to infer remittances amount
df_gdp = pd.read_pickle("c:\\data\\economic\\gdp\\annual_gdp_per_capita_splined.pkl")
df = df.merge(df_gdp, on=['destination', 'date'], how='left')

## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags.pkl")

####parameters

params = [2.725302695640765, -9.769960496731258, 8.310039749519659,
            0.1788188275520765, 0.21924650014050806,-0.7500114294211869,
            0.3856959277016759, 0.1333609093157307]
# params = [1.89, -9.466, 7, 0.27, 0.39, -2, -0.84, 0.11]
# params = [1.8979563003878646, -7.711374890168264,9.728430227336691,
#           0.19573951150137103,0.20461772706094322,-1.3942682139559333,
#           -1.6491738989620697, 0.1]

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
def simulate_remittances(df_countries, height, shape, shift, rem_pct, disasters = True):

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

    df_countries['theta'] = constant + (param_nta * (df_countries['nta'])) \
                            + (param_asy * df_countries['asymmetry']) + (param_gdp * df_countries['gdp_diff_norm']) \
                            + (df_countries['tot_score'])
    df_countries.loc[df_countries.nta == 0, 'theta'] == 0
    df_countries['probability'] = 1 / (1 + np.exp(-df_countries["theta"]))
    df_countries['sim_senders'] = (df_countries['probability'] * df_countries['n_people']).astype(int)
    df_countries['sim_remittances'] = df_countries['sim_senders'] * df_countries['rem_amount']

    remittance_per_period = df_countries.groupby(['date', 'origin', 'destination'])[['sim_remittances', 'sim_senders']].sum().reset_index()

    return remittance_per_period

def quick_check_mex_(df_countries, height, shape, shift, rem_pct, plot = True, disasters = True):

    df_countries = df_countries[(df_countries.origin == "Mexico") & (df_countries.destination == "USA")]
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

    df_countries['theta'] = constant + (param_nta * (df_countries['nta'])) \
                            + (param_asy * df_countries['asymmetry']) + (param_gdp * df_countries['gdp_diff_norm']) \
                            + (df_countries['tot_score'])
    df_countries.loc[df_countries.nta == 0, 'theta'] == 0
    df_countries['probability'] = 1 / (1 + np.exp(-df_countries["theta"]))
    df_countries['sim_senders'] = (df_countries['probability'] * df_countries['n_people']).astype(int)
    df_countries['sim_remittances'] = df_countries['sim_senders'] * df_countries['rem_amount']

    remittance_per_period = df_countries.groupby(['date', 'origin', 'destination'])[['sim_remittances', 'sim_senders']].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_mex, on = ["date", "origin", "destination"], how = 'left')

    if plot:
        goodness_of_fit_results(remittance_per_period)

    return remittance_per_period

sim_mexico = quick_check_mex_(df, height, shape, shift, rem_pct)

##### run simulations
# WITH DISASTERS
biggest_countries = df[["origin", "n_people"]].groupby('origin').sum().sort_values('n_people', ascending = False)[:20].index.tolist()

print("simulating disasters ....")
df_results = simulate_remittances(df, height, shape, shift, rem_pct, disasters = True)
df_results.to_pickle("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_with_disasters_2307.pkl")
# WITHOUT
print("simulating NO disasters ...")
df_results = simulate_remittances(df, height, shape, shift, rem_pct, disasters = False)
df_results.to_pickle("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_without_disasters_2307.pkl")

#################
# plot
#################

df_with = pd.read_pickle("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_with_disasters_2307.pkl")
df_without = pd.read_pickle("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_without_disasters_2307.pkl")

##########
# save df_with aggregate
df_all = df_with[["date", "origin", "destination", "sim_remittances"]].merge(
    df_without[["date", "origin", "destination", "sim_remittances"]],
    on = ["date", "origin", "destination"], suffixes = ("_with", "_without"))
df_all['year'] = df_all.date.dt.year
df_all_group = (df_all[["year", "origin", "destination", "sim_remittances_with", "sim_remittances_without"]].
                groupby(["year", "origin", "destination"]).sum().reset_index())
df_all_group.shape
df_all_group.rename(columns = {'origin' : 'receiver', 'destination' : 'sender'}, inplace = True)
df_results.to_csv("C:\\git-projects\\csh_remittances\\general\\results_plots\\yearly_flows.csv", index = False)

##########

df_with['quarter'] = df_with.date.dt.to_period('Q').dt.to_timestamp()
df_without['quarter'] = df_without.date.dt.to_period('Q').dt.to_timestamp()

df_with_quarter = df_with[['quarter', 'sim_remittances']].groupby("quarter").sum().reset_index()
df_with_quarter['sim_remittances'] /= 1e9
df_with_quarter.set_index('quarter', inplace = True)
df_without_quarter = df_without[['quarter', 'sim_remittances']].groupby("quarter").sum().reset_index()
df_without_quarter['sim_remittances'] /= 1e9
df_without_quarter.set_index('quarter', inplace = True)

fig, ax = plt.subplots(figsize = (9,6))

plt.plot(df_with_quarter.iloc[:-1], label = "With disasters")
plt.plot(df_without_quarter[:-1], label = "Without disasters")
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

#############################
# Plots by precise date
##############################


df_with_date = df_with[['date', 'sim_remittances']].groupby("date").sum().reset_index()
df_with_date['sim_remittances'] /= 1e9
df_with_date.set_index('date', inplace = True)
df_without_date = df_without[['date', 'sim_remittances']].groupby("date").sum().reset_index()
df_without_date['sim_remittances'] /= 1e9
df_without_date.set_index('date', inplace = True)

df_all_group = df_with_date.merge(df_without_date, on = "date", suffixes = ("_with", "_without"))
df_all_group['difference'] = df_all_group['sim_remittances_with'] - df_all_group['sim_remittances_without']

fig, ax = plt.subplots(figsize = (9,6))

plt.plot(df_with_date.iloc[:-1], label = "With disasters")
plt.plot(df_without_date[:-1], label = "Without disasters")
plt.legend()
plt.grid(True)
plt.show(block = True)

fig, ax = plt.subplots(figsize = (9,6))

plt.plot(df_all_group['difference'])
plt.grid(True)
plt.show(block = True)

############################
# Analysis
###########################

df_in_country = pd.read_excel(data_folder + "inward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y")
df_in_country.rename(columns = {"Remittance inflows (US$ million)": "country"}, inplace=True)
df_in_country = pd.melt(df_in_country, id_vars=['country'], value_vars=df_in_country.columns.tolist()[1:])
df_in_country.rename(columns = {"variable": "year", "value" : "inflow"}, inplace=True)
df_in_country.year = df_in_country.year.astype('int')
df_in_country = df_in_country[(df_in_country.year > 2009) & (df_in_country.year < 2020)]
df_in_country['inflow'] /= 1_000
df_in_country = df_in_country[["country", "inflow"]].groupby('country').sum().reset_index()
df_in_country["country"] = df_in_country.country.str.strip().map(dict_names)

# which countries got the most because of disasters
df_all = df_with[["date", "quarter", "origin", "destination", "sim_remittances"]].merge(
    df_without[["date", "quarter", "origin", "destination", "sim_remittances"]],
    on = ["date", "quarter", "origin", "destination"], suffixes = ("_with", "_without"))
df_all['sim_remittances_without'] /= 1e9
df_all['sim_remittances_with'] /= 1e9

df_by_country_dest = (df_all[["origin", "sim_remittances_with", "sim_remittances_without"]]
                      .groupby('origin').sum().reset_index())
df_by_country_dest.rename(columns = {'origin' : 'country'}, inplace = True)
df_by_country_dest = df_by_country_dest.merge(df_in_country, on = 'country', how = 'left')
df_by_country_dest['difference'] = df_by_country_dest['sim_remittances_with'] - df_by_country_dest['sim_remittances_without']
df_by_country_dest['pct_difference'] = round(100 * df_by_country_dest['difference'] / df_by_country_dest['sim_remittances_without'],2)
df_by_country_dest.sort_values('difference', ascending = False, inplace = True)

from tabulate import tabulate
print(tabulate(df_by_country_dest.head(10), headers='keys', tablefmt='pretty'))

from sklearn.linear_model import LinearRegression

# Filter out non-positive values for log-log plot
df_filtered = df_by_country_dest[
    (df_by_country_dest['inflow'] > 1) &
    (df_by_country_dest['sim_remittances_with'] > 1)
]

# Log-transform data
x_log = np.log10(df_filtered['inflow'])
y_log = np.log10(df_filtered['sim_remittances_with'])

# Fit linear regression in log-log space
reg = LinearRegression().fit(x_log.values.reshape(-1, 1), y_log.values)
r2 = reg.score(x_log.values.reshape(-1, 1), y_log.values)

# Create scatter plot
fig = px.scatter(
    df_filtered,
    x='inflow',
    y='sim_remittances_with',
    color='country',  # change if your column name is different
    log_x=True,
    log_y=True,
    labels={
        'inflow': 'Est remittances world bank (billion)',
        'sim_remittances_with': 'Sim remittances my model (billion)'
    },
    width=1200,
    height=800,
    template='simple_white'
)

# Add 45-degree y = x line
lims = [
    min(df_filtered['inflow'].min(), df_filtered['sim_remittances_with'].min()),
    max(df_filtered['inflow'].max(), df_filtered['sim_remittances_with'].max())
]
fig.add_trace(go.Scatter(
    x=lims,
    y=lims,
    mode='lines',
    name='y = x',
    line=dict(color='black', dash='solid')
))

# Add R² annotation
fig.add_annotation(
    xref='paper', yref='paper',
    x=0.05, y=0.95,
    showarrow=False,
    text=f"R² (log-log): {r2:.3f}",
    font=dict(size=14)
)

# Add gridlines explicitly
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')

# Final layout tweaks
fig.update_layout(
    legend_title_text='Country',
    title='Comparison of remittances inflow (KNOMAD vs. our model, total inflow from 2010 to 2020)'
)
fig.write_html("C:\\git-projects\\csh_remittances\\plots\\for_paper\\latest_results\\comparison_KNOMAD.html")
fig.show()

#################################
# Maps
#################################
from geodatasets import get_path
import geopandas as gpd
world = gpd.read_file(get_path("naturalearth.land"))
world['country'] = world['country'].map(dict_names)

# Merge with GeoDataFrame
merged = world.merge(df_filtered, how='left', left_on='name', right_on='country')

# Plot
merged.plot(column='remittances', cmap='Blues', legend=True, figsize=(12, 8))
plt.title('Remittances Received by Country')
plt.axis('off')
plt.show()

fig = px.choropleth(df_filtered,
                    locations='country',
                    locationmode='country names',
                    color='difference',
                    color_continuous_scale='Greens',
                    title='Remittances received by country (response to disasters)')
fig.show()

fig = px.choropleth(df_filtered,
                    locations='country',
                    locationmode='country names',
                    color='sim_remittances_with',
                    color_continuous_scale='Greens',
                    title='Remittances received by country (totals)')
fig.show()

def plot_one_receiver_senders(country):
    df_dest = (df_with[df_with.origin == country][["date", "origin", "destination", "sim_remittances"]]
                               .groupby(['date', 'origin', 'destination']).sum().reset_index().sort_values(['date', 'sim_remittances'],
                                                                                            ascending=False))

    fig = px.scatter(df_dest, x='date', y='sim_remittances', color='destination',
                     height=600, width=1200, template='simple_white',
                     labels={'date': 'Date', 'sim_remittances': 'Simulated remittances (billions)'}).update_traces(
        mode='lines')
    # Add gridlines explicitly
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    # Final layout tweaks
    fig.update_layout(legend_title_text='Country',
                      title=f'Timeseries of remittances inflow to {country}')
    fig.show()

plot_one_receiver_senders("United Kingdom")
plot_one_receiver_senders("Mexico")
plot_one_receiver_senders("Bangladesh")

