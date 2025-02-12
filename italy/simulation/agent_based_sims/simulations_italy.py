
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.special import expit
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
from italy.simulation.func.goodness_of_fit import plot_remittances_senders, plot_all_results_log, goodness_of_fit_results

outfolder = "C:\\git-projects\\csh_remittances\\italy\\plots\\plots_for_paper\\model_results\\"
fixed_vars = ['agent_id', 'country', 'sex']

## nta
nta = pd.read_excel("c:\\data\\economic\\nta\\NTA profiles.xlsx", sheet_name="italy").T
nta.columns = nta.iloc[0]
nta = nta.iloc[1:]
nta.reset_index(names='age', inplace = True)
nta = nta[['age', 'Support Ratio']].rename(columns = {'Support Ratio' : 'nta'})
nta.nta=nta.nta * 0.7

## agents
df_ag = pd.read_pickle("c:\\data\\population\\italy\\simulated_migrants_populations_2008_2022.pkl")
df_age = df_ag[fixed_vars + [x for x in df_ag.columns if '_age' in x]]
df_age = pd.melt(df_age, id_vars=fixed_vars, value_vars=df_age.columns[3:],
             value_name='age', var_name='year')
df_age['year'] = df_age['year'].apply(lambda x: int(x[:4]))
df_stay = df_ag[fixed_vars + [x for x in df_ag.columns if '_yrs' in x]]
df_stay = pd.melt(df_stay, id_vars=fixed_vars, value_vars=df_stay.columns[3:],
             value_name='yrs_stay', var_name='year')
df_stay['year'] = df_stay['year'].apply(lambda x: int(x[:4]))
df_prob = df_ag[fixed_vars + [x for x in df_ag.columns if '_fam' in x]]
df_prob = pd.melt(df_prob, id_vars=fixed_vars, value_vars=df_prob.columns[3:],
             value_name='fam_prob', var_name='year')
df_prob['year'] = df_prob['year'].apply(lambda x: int(x[:4]))
df_ag_long = df_age.merge(df_stay, on = fixed_vars + ['year'], how = 'left').merge(df_prob, on = fixed_vars + ['year'], how = 'left')
df_ag_long = df_ag_long.merge(nta, on = 'age', how = 'left')

###
df_ag_long['theta'] =  param_nta * df_ag_long['nta'] + param_stay * df_ag_long['yrs_stay'] + param_fam * df_ag_long['fam_prob']
df_ag_long['theta'] = pd.to_numeric(df_ag_long['theta'], errors='coerce')
df_ag_long['prob'] = 1 / (1 + np.exp(-df_ag_long['theta']))
#
df_ag_long['theta'].hist()
plt.show(block = True)
#
df_ag_long['prob'].hist()
plt.show(block = True)
plt.plot(df_ag_long['prob'].sort_values().tolist())
plt.show(block = True)
###
## remittances
df = pd.read_parquet("C:\\Data\\my_datasets\\italy\\simulation_data.parquet")
df['date'] = pd.to_datetime(df.date)

#parameters
param_nta = 6
param_stay = -0.3
param_fam = -1
rem_amount = 1500

def simulate_one_country_no_disasters(country, plot, disable_progress = False):
    rem_country = df[(df.country == country) & (df.age_group == 'Less than 5 years') & (df.sex == 'male')].copy()

    df_country = df_ag_long[df_ag_long.country == country].copy()
    df_country['theta'] = param_nta * df_country['nta'] + param_stay * df_country['yrs_stay'] + param_fam * df_country['fam_prob']
    df_country['theta'] = pd.to_numeric(df_country['theta'], errors='coerce')
    df_country['prob'] = 1 / (1 + np.exp(-df_country['theta']))

    senders = []
    for date in tqdm(rem_country.date, disable=disable_progress):
        probs = df_country[(df_country.year == date.year) & (~df_country.prob.isna())]['prob'].tolist()
        senders.append(np.random.binomial(1, probs).sum())
    res = rem_country[['date', 'remittances']].copy()
    res['simulated_senders'] = senders

    if plot:
        plot_remittances_senders(res)
    return res

nep_res = simulate_one_country_no_disasters("Nepal", True)

def simulate_all_countries():
    res = pd.DataFrame([])
    for country in tqdm(df_ag_long.country.unique()):
        country_res = simulate_one_country_no_disasters(country, plot = False , disable_progress = True)
        country_res['country'] = country
        res = pd.concat([res, country_res])
    return res

df_res = simulate_all_countries()
df_res['sim_remittances'] = df_res.simulated_senders * rem_amount
df_res_small = df_res[df_res['remittances'] > 10_000]

plot_all_results_log(df_res_small)
goodness_of_fit_results(df_res_small)


