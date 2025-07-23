
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
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors_2207_TRAIN.pkl"
df = pd.read_pickle(diasporas_file)
# GDP data file
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx").rename(columns = {'country' : 'destination'})#.groupby('country').mean().reset_index().rename(columns = {'country' : 'origin'}).drop(columns = 'year')
# EMDAT file
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags.pkl")
# our model results
df_with = pd.read_pickle("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_with_disasters_2207_TRAIN.pkl")
df_without = pd.read_pickle("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_without_disasters_2207_TRAIN.pkl")
# KNOMAD inflows
data_folder = "C:\\git-projects\\csh_remittances\\data_downloads\\data\\"
df_in = pd.read_excel(data_folder + "inward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y")
df_in.rename(columns = {"Remittance inflows (US$ million)": "country"}, inplace=True)
df_in = pd.melt(df_in, id_vars=['country'], value_vars=df_in.columns.tolist()[1:])
df_in.rename(columns = {"variable": "year", "value" : "inflow"}, inplace=True)
df_in.year = df_in.year.astype('int')
df_in = df_in[(df_in.year > 2009) & (df_in.year < 2020)]
df_in['inflow'] /= 1_000
df_in = df_in[["year", "inflow"]].groupby('year').sum()
# income and region classification
income_class = pd.read_excel("C:\\Data\\economic\\income_classification_countries_wb.xlsx")
income_class['country'] = income_class.country.map(dict_names)
income_class = income_class[["country", "group", "Region"]]

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

######################
df_with['year'] = df_with.date.dt.year
df_without['year'] = df_without.date.dt.year
df_with_period = df_with[['year', 'sim_remittances']].groupby("year").sum().reset_index()
df_with_period['sim_remittances'] /= 1e9
df_with_period.set_index('year', inplace = True)
df_without_period = df_without[['year', 'sim_remittances']].groupby("year").sum().reset_index()
df_without_period['sim_remittances'] /= 1e9
df_without_period.set_index('year', inplace = True)

fig, ax = plt.subplots(figsize = (9,6))
plt.plot(df_with_period.iloc[:-1], label = "Our model with disasters")
plt.plot(df_without_period[:-1], label = "Our model without disasters")
plt.plot(df_in, label = "World Bank inflow estimate")
plt.ylabel("Total remittances (bn US dollars)")
plt.title("Total remittances flows around the world")
plt.legend()
plt.grid(True)
plt.show(block = True)

# which countries got the most because of disasters
df_all = df_with[["date", "origin", "destination", "sim_remittances"]].merge(
    df_without[["date", "origin", "destination", "sim_remittances"]],
    on = ["date", "origin", "destination"], suffixes = ("_with", "_without"))
df_all['sim_remittances_without'] /= 1e9
df_all['sim_remittances_with'] /= 1e9

df_by_country_dest = (df_all[["origin", "sim_remittances_with", "sim_remittances_without"]]
                      .groupby('origin').sum().reset_index())
df_by_country_dest.rename(columns = {'origin' : 'country'}, inplace = True)
df_by_country_dest = df_by_country_dest.merge(df_in_country, on = 'country', how = 'left')
df_by_country_dest['difference'] = df_by_country_dest['sim_remittances_with'] - df_by_country_dest['sim_remittances_without']
df_by_country_dest['pct_difference'] = round(100 * df_by_country_dest['difference'] / df_by_country_dest['sim_remittances_without'],2)
df_by_country_dest.sort_values('difference', ascending = False, inplace = True)

# Filter out non-positive values for log-log plot
df_filtered = df_by_country_dest[
    (df_by_country_dest['inflow'] > 1) &
    (df_by_country_dest['sim_remittances_with'] > 1)]
x_log = np.log10(df_filtered['inflow'])
y_log = np.log10(df_filtered['sim_remittances_with'])

# Fit linear regression in log-log space
reg = LinearRegression().fit(x_log.values.reshape(-1, 1), y_log.values)
r2 = reg.score(x_log.values.reshape(-1, 1), y_log.values)

# Create scatter plot
fig = px.scatter(df_filtered,x='inflow',y='sim_remittances_with',
    color='country', log_x=True, log_y=True,
    labels={'inflow': 'Est remittances world bank (billion)',
        'sim_remittances_with': 'Sim remittances my model (billion)'},
    height = 500, width = 900, template='simple_white')

# Add 45-degree y = x line
lims = [min(df_filtered['inflow'].min(), df_filtered['sim_remittances_with'].min()),
        max(df_filtered['inflow'].max(), df_filtered['sim_remittances_with'].max())]
fig.add_trace(go.Scatter(x=lims,y=lims,
    mode='lines',name='y = x', line=dict(color='black', dash='solid')))

# Add R² annotation
fig.add_annotation(xref='paper', yref='paper', x=0.05, y=0.95,
    showarrow=False, text=f"R² (log-log): {r2:.3f}", font=dict(size=14))

# Add gridlines explicitly
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')

# Final layout tweaks
fig.update_layout( legend_title_text='Country',
    title='Comparison of remittances inflow (KNOMAD vs. our model, total inflow from 2010 to 2020)')
fig.show()