
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

df_mig_pop = pd.read_excel("C:\\Data\\migration\\bilateral_migration_undesa.xlsx",
                           sheet_name='Table 1', skiprows=10).iloc[:, 1:]
df_mig_pop.columns = ['dest','coverage','dtype','code','origin', 'codeor',
                      1990,1995,2000,2005,2010,2015,2020,2024,
                      '1990_m','1995_m','2000_m','2005_m','2010_m','2015_m','2020_m','2024_m',
                      '1990_f','1995_f','2000_f','2005_f','2010_f','2015_f','2020_f','2024_f']

df_ita = df_mig_pop[df_mig_pop.dest == "Italy"][['origin',1990,1995,2000,2005,2010,2015,2020,2024]]
df_ita.origin = df_ita.origin.apply(lambda x: x.replace("*", ""))
df_ita.origin = df_ita.origin.map(dict_names)
df_ita.dropna(inplace = True)
df_ita = pd.melt(df_ita, id_vars = 'origin', value_vars=[1990,1995,2000,2005,2010,2015,2020,2024],
                 var_name='year', value_name='population')
df_ita['year'] = pd.to_datetime(df_ita['year'], format = '%Y')
start_date, end_date = df_ita['year'].min(), df_ita['year'].max()
yearly_dates = pd.date_range(start=start_date, end=end_date, freq='Y')
yearly_times = (yearly_dates - start_date).days
df_ita['time'] = (df_ita['year'] - df_ita['year'].min()).dt.days

cols = ['year', 'origin', 'population']
df_ita_int = pd.DataFrame()

for country in tqdm(df_ita.origin.unique()):
    data = [yearly_dates, [country] * len(yearly_dates)]
    for col in cols[2:]:
        cs = CubicSpline(df_ita[df_ita.origin == country]['time'],
                         df_ita[df_ita.origin == country][col])
        vals = cs(yearly_times)
        data.append(vals)
    dict_country = dict(zip(cols, data))
    country_df = pd.DataFrame(dict_country)
    df_ita_int = pd.concat([df_ita_int, country_df])
df_ita_int["year"] = df_ita_int["year"].dt.year
df_ita_int.rename(columns={'origin' : 'country'}, inplace = True)
df_ita_int['population'] = df_ita_int['population'].astype(int)

##################
df_rem = pd.read_csv('c:\\data\\remittances\\italy\\monthly_splined_remittances.csv')
df_rem['date'] = pd.to_datetime(df_rem['date'])
df_rem['year'] = df_rem['date'].dt.year

df = df_rem.merge(df_ita_int, on = ['country', 'year'], how = 'left')
df.dropna(inplace = True)

df_nat = pd.read_csv("C:\\Data\\my_datasets\\weekly_remittances\\weekly_disasters.csv")
df_nat["week_start"] = pd.to_datetime(df_nat["week_start"])
df_nat["year"] = df_nat.week_start.dt.year
df_pop_country = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")
df_nat = df_nat.merge(df_pop_country, on = ['country', 'year'], how = 'left')
df_nat['total_affected'] = 100 * df_nat['total_affected'] / df_nat["population"]
df_nat = df_nat[["week_start", "total_affected", "total_damage", "country", "type"]]
df_nat_monthly = (
    df_nat.groupby(['country', 'type', pd.Grouper(key='week_start', freq='M')])
    .agg({'total_affected': 'sum', 'total_damage': 'sum'})
    .reset_index()
    .rename(columns={'week_start': 'date'}))

df = df.merge(df_nat_monthly[['country','date', 'total_affected', 'total_damage']],
              on = ["country", "date"], how = 'left')
df.fillna(0, inplace = True)

## simulation
# Parameters
base_prob = 0.8  # Base probability of sending remittances
max_boost = 0.5  # Maximum possible increase in probability
fixed_amount = 100  # Fixed amount sent per migrant
lag_months = 3  # Number of months to consider for disaster effect
def simulate_remittances(df, base_prob, max_boost, fixed_amount, lag_months):
    df = df.sort_values(by=['country', 'date']).copy()

    # Compute rolling sum of `total_affected` for the past (lag_months + 1) months
    df['affected_lagged'] = df.groupby('country')['total_affected'].transform(lambda x: x.rolling(lag_months + 1, min_periods=1).sum())

    # Normalize the effect (scaling between 0 and max_boost)
    max_affected = df['affected_lagged'].max()
    if max_affected > 0:
        df['disaster_boost'] = (df['affected_lagged'] / max_affected) * max_boost
    else:
        df['disaster_boost'] = 0  # No disasters at all

    # Compute probability of sending remittances
    df['prob'] = base_prob + df['disaster_boost']

    # Simulate remittances
    df['simulated_remittances'] = df.apply(lambda row: np.random.binomial(row['population'], min(row['prob'], 1)) * fixed_amount, axis=1)

    return df

# Run the simulation
df_sim = simulate_remittances(df, base_prob, max_boost, fixed_amount, lag_months)


# Display the results
fig = px.scatter(df_sim, x = 'remittances', y='simulated_remittances',
                 color = 'country', log_x = True, log_y = True)
fig.show()

def country_results(country):
    df_country = df_sim[df_sim.country == country]
    fig = px.line(df_country, x='date', y='simulated_remittances')
    # fig.add_trace(go.Scatter(x = df_country.date, y = df_country.remittances))
    fig.show()

country_results('Bangladesh')