import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = 'browser'
from utils import italy_close_countries
from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)

outfolder = "C:\\git-projects\\csh_remittances\\italy\\plots\\plots_for_paper\\model_results\\"
fixed_vars = ['agent_id', 'country', 'sex']

#parameters
param_nta = 1
param_stay = -0.1
param_fam = -2
param_gdp = -2
rem_amount = 3000
params = [param_nta, param_stay, param_fam, param_gdp, rem_amount]

### load nta
nta = pd.read_excel("c:\\data\\economic\\nta\\NTA profiles.xlsx", sheet_name="italy").T
nta.columns = nta.iloc[0]
nta = nta.iloc[1:]
nta.reset_index(names='age', inplace = True)
nta = nta[['age', 'Support Ratio']].rename(columns = {'Support Ratio' : 'nta'})

## load agents
df_ag = pd.read_pickle("c:\\data\\population\\italy\\simulated_migrants_populations_2008_2022.pkl")
df_age = df_ag[fixed_vars + [x for x in df_ag.columns if '_age' in x]]
df_age = pd.melt(df_age, id_vars=fixed_vars, value_vars=df_age.columns[3:],
             value_name='age', var_name='year')
df_age['year'] = df_age['year'].apply(lambda x: int(x[:4]))
df_stay = df_ag[fixed_vars + [x for x in df_ag.columns if '_yrs' in x]]
df_stay = pd.melt(df_stay, id_vars=fixed_vars, value_vars=df_stay.columns[3:],
             value_name='yrs_stay', var_name='year')
df_stay['year'] = df_stay['year'].apply(lambda x: int(x[:4]))
df_stay_group = df_stay[['country', 'year', 'yrs_stay']].groupby(['country', 'year']).mean().reset_index()
df_stay_group['yrs_stay_eff'] = 1 / (1 + np.exp(-(df_stay_group['yrs_stay'] * param_stay + 1)))
# fig = px.line(df_stay_group, x = 'year', y = 'yrs_stay_eff', color = 'country')
# fig.show()

df_prob = df_ag[fixed_vars + [x for x in df_ag.columns if '_fam' in x]]
df_prob = pd.melt(df_prob, id_vars=fixed_vars, value_vars=df_prob.columns[3:],
             value_name='fam_prob', var_name='year')
df_prob['year'] = df_prob['year'].apply(lambda x: int(x[:4]))
df_prob_group = df_prob[['country', 'year', 'fam_prob']].groupby(['country', 'year']).mean().reset_index()
df_prob_group['fam_prob_eff'] = 1 / (1 + np.exp(-(df_prob_group['fam_prob'] * param_fam + 10)))
# fig = px.line(df_prob_group, x = 'year', y = 'fam_prob_eff', color = 'country')
# fig.show()

df_ag_long = df_age.merge(df_stay, on = fixed_vars + ['year'], how = 'left').merge(df_prob, on = fixed_vars + ['year'], how = 'left')
df_ag_long = df_ag_long.merge(nta, on = 'age', how = 'left')

## load remittances
df = pd.read_parquet("C:\\Data\\my_datasets\\italy\\simulation_data.parquet")
df['date'] = pd.to_datetime(df.date)
df.sort_values(['country', 'date'], inplace = True)
df_rem_group = df[~df[["date", "country"]].duplicated()][["date", "country", "remittances", "gdp_per_capita", "delta_gdp"]]
df_pop_group = df[["date", "country", "population"]].groupby(["date", "country"]).sum().reset_index()
df_rem_group = df_rem_group.merge(df_pop_group, on = ["date", "country"], how = 'left')
df_rem_group['exp_pop'] = df_rem_group['remittances'] / rem_amount
df_rem_group['pct_sending'] = df_rem_group['exp_pop'] / df_rem_group['population']
df_rem_group['year'] = df_rem_group["date"].dt.year

# country_avg = df_rem_group[['country', 'pct_sending']].groupby('country').mean()
# country_avg['pct_sending'].hist(bins = 50)
# plt.show(block=True)

df_year_gdp = df_rem_group[['country', 'year', "gdp_per_capita", "delta_gdp"]].groupby(['country', 'year']).mean().reset_index()
df_ag_long = df_ag_long.merge(df_year_gdp, on = ['country', 'year'], how = 'left')

### normalise delta gdp
df_ag_long["delta_gdp_norm"] = df_ag_long.delta_gdp / abs(df_ag_long.delta_gdp.min())
gdp_group = df_ag_long[['country', 'year', 'delta_gdp_norm']].groupby(['country', 'year']).mean().reset_index()
gdp_group['gdp_eff'] = 1 / (1 + np.exp(-(gdp_group['delta_gdp_norm'] * param_gdp)))

# fig = px.bar(gdp_group[gdp_group.year == 2015], x = 'country', y = 'delta_gdp_norm')
# fig.show()

# fig = px.line(gdp_group, x = 'year', y = 'gdp_eff', color = 'country')
# fig.show()

#group nta
nta_group = df_ag_long[['country', 'year', 'nta']].groupby(['country', 'year']).mean().reset_index()