
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from pandas.tseries.offsets import MonthEnd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# dictionary of country names
import json
dict_names = {}
with open('c:\\data\\general\\countries_dict.txt',
          encoding='utf-8') as f:
    data = f.read()
js = json.loads(data)
for k,v in js.items():
    for x in v:
        dict_names[x] = k

# file with diasporas characteristics
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors_splined.pkl"
df = pd.read_pickle(diasporas_file)
# GDP data file
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx").rename(columns = {'country' : 'destination'})#.groupby('country').mean().reset_index().rename(columns = {'country' : 'origin'}).drop(columns = 'year')
# EMDAT file
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags_NEW.pkl")
# our model results
df_with = pd.read_pickle("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_with_disasters.pkl")

#####
# agg by country dest
df_by_country_dest_time = (df_with[["date", "origin", "sim_remittances"]]
                      .groupby(['date', 'origin']).sum().reset_index().sort_values(['date', 'sim_remittances'], ascending = False))

fig = px.scatter(df_by_country_dest_time, x = 'date', y='sim_remittances', color = 'origin',
                height = 600, width = 1200, template='simple_white',
                labels={'date': 'Date', 'sim_remittances': 'Simulated remittances (billions)'}).update_traces(mode='lines')
# Add gridlines explicitly
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')

# Final layout tweaks
fig.update_layout( legend_title_text='Country',
    title='Timeseries of remittances inflow according to our model')
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
