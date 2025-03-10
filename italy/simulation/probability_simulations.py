

import pandas as pd
import numpy as np
from utils import dict_names
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = 'browser'
from plotly.subplots import make_subplots
import re
import seaborn as sns
import time
from italy.simulation.func.goodness_of_fit import goodness_of_fit_results, plot_all_results_log, plot_lines
sns.set_style('whitegrid')

#remittances and disasters
df_rem = pd.read_csv('c:\\data\\remittances\\italy\\monthly_splined_remittances.csv')
df_rem['date'] = pd.to_datetime(df_rem['date'])
df_rem['year'] = df_rem['date'].dt.year

df_nat = pd.read_csv("C:\\Data\\my_datasets\\weekly_remittances\\weekly_disasters.csv")
df_nat["week_start"] = pd.to_datetime(df_nat["week_start"])
df_nat["year"] = df_nat.week_start.dt.year
df_pop_country = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")
df_nat = df_nat.merge(df_pop_country, on = ['country', 'year'], how = 'left')
df_nat['total_affected'] = 100 * df_nat['total_affected'] / df_nat["population"]
df_nat = df_nat[["week_start", "total_affected", "total_damage", "country", "type"]]
df_nat_monthly = (
    df_nat.groupby(['country', pd.Grouper(key='week_start', freq='M')])
    .agg({'total_affected': 'sum', 'total_damage': 'sum'})
    .reset_index()
    .rename(columns={'week_start': 'date'}))

df_rem = df_rem.merge(df_nat_monthly[['country','date', 'total_affected', 'total_damage']],
              on = ["country", "date"], how = 'left')
df_rem.fillna(0, inplace = True)

# population and national transfers account
df = pd.read_csv('c:\\data\\migration\\italy\\estimated_stocks_new.csv')
df['age'] = df.age_group.astype(str).apply(lambda x: np.mean(list(map(int, re.findall(r'\d+', x)))))
df.loc[df.age == 5, 'age'] = 2.5
df['age'] = df.age.astype(int)
df.sort_values(['citizenship', 'year', 'age'], inplace = True)

nta = pd.read_excel("c:\\data\\economic\\nta\\NTA profiles.xlsx", sheet_name="italy").T
nta.columns = nta.iloc[0]
nta = nta.iloc[1:]
nta.reset_index(names='age', inplace = True)
nta = nta[['age', 'Support Ratio']].rename(columns = {'Support Ratio' : 'nta'})
nta.nta=(nta.nta-nta.nta.min())/(nta.nta.max()-nta.nta.min()) - 0.15
nta.loc[nta.nta <0, 'nta'] = 0

df = df.merge(nta, on='age', how = 'left')
df.rename(columns = {'citizenship' : 'country', 'count' : 'population'}, inplace = True)
df = df.merge(df_rem, on = ['country', 'year'], how = 'left')
df.dropna(inplace = True)
df.loc[df.population < 0, 'population'] = 0

# shift disasters
df['date'] = pd.to_datetime(df['date'])
# Create shifted columns using proper datetime handling
for shift in tqdm([1, 2, 3, 4]):
    g = df.groupby(['country', 'age_group', 'sex'], group_keys=False)
    g =  g.apply(lambda x: x.set_index('date')['total_affected']
                 .shift(shift).reset_index(drop=True)).fillna(0)
    df[f'ta_{shift}'] = g.tolist()

##############
# individual probability process
##############
male_boost = 0.05 #(5%)
d_boost_0 = 1
d_boost_1 = 2
d_boost_2 = 3
d_boost_3 = 2.5
d_boost_4 = 1.5
fixed_rem_amount = 250
def probability_single_country(country, plot = True):
    df_ = df[(df.country == country)].copy()

    df_['prob'] = df_.nta
    df_.loc[(df_.sex == 'male') & (df_.nta > 0), 'prob'] += male_boost

    ## disasters effect
    df_.loc[df_['prob'] > 0, 'prob'] += df_['total_affected'] * d_boost_0
    df_.loc[df_['prob'] > 0, 'prob'] += df_['ta_1'] * d_boost_1
    df_.loc[df_['prob'] > 0, 'prob'] += df_['ta_2'] * d_boost_2
    df_.loc[df_['prob'] > 0, 'prob'] += df_['ta_3'] * d_boost_3
    df_.loc[df_['prob'] > 0, 'prob'] += df_['ta_4'] * d_boost_4

    df_.loc[df_['prob'] < 0, 'prob'] = 0
    df_.loc[df_['prob'] > 1, 'prob'] = 1

    df_['simulated_senders'] = df_.apply(lambda row: np.random.binomial(row['population'], min(row['prob'], 1)), axis=1)

    df_plot = df_[['date', 'simulated_senders', 'remittances', 'population']].groupby('date').agg({
        'simulated_senders' : 'sum', 'remittances': 'mean', 'population': 'sum'
    }).reset_index()
    df_plot['sim_remittances'] = df_plot.simulated_senders * fixed_rem_amount
    df_plot['error'] = np.abs(df_plot['remittances'] - df_plot['sim_remittances'])
    if plot:
        plot_lines(df_plot)
    return df_plot

probability_single_country('Bangladesh')

def probability_all_countries(df):
    df['prob'] = df.nta
    df.loc[(df.sex == 'male') & (df.nta > 0), 'prob'] += male_boost

    ## disasters effect
    df.loc[df['prob'] > 0, 'prob'] += df['total_affected'] * d_boost_0
    df.loc[df['prob'] > 0, 'prob'] += df['ta_1'] * d_boost_1
    df.loc[df['prob'] > 0, 'prob'] += df['ta_2'] * d_boost_2
    df.loc[df['prob'] > 0, 'prob'] += df['ta_3'] * d_boost_3
    df.loc[df['prob'] > 0, 'prob'] += df['ta_4'] * d_boost_4

    df.loc[df['prob'] < 0, 'prob'] = 0
    df.loc[df['prob'] > 1, 'prob'] = 1

    df['simulated_senders'] = df.apply(lambda row: np.random.binomial(row['population'], min(row['prob'], 1)), axis=1)

    df_plot = df[['date', 'country', 'simulated_senders', 'remittances', 'population']].groupby(['date', 'country']).agg({
        'simulated_senders': 'sum', 'remittances': 'mean', 'population': 'sum'
    }).reset_index()
    df_plot['sim_remittances'] = df_plot.simulated_senders * fixed_rem_amount
    df_plot['error'] = np.abs(df_plot['remittances'] - df_plot['sim_remittances'])

    return df_plot

start = time.time()
df_results = probability_all_countries(df[df.country.isin(df.country.unique().tolist())].copy())
end = time.time()
print(f"Second elapsed: {np.round(end - start,2)}")

plot_all_results_log(df_results)
goodness_of_fit_results(df_results)






