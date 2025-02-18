
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
from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)

outfolder = "C:\\git-projects\\csh_remittances\\italy\\plots\\plots_for_paper\\model_results\\"
fixed_vars = ['agent_id', 'country', 'sex']

param_nta = 40
param_stay = -0.1
param_fam = -2
param_gdp = -2
rem_amount = 5000

## nta
nta = pd.read_excel("c:\\data\\economic\\nta\\NTA profiles.xlsx", sheet_name="italy").T
nta.columns = nta.iloc[0]
nta = nta.iloc[1:]
nta.reset_index(names='age', inplace = True)
nta = nta[['age', 'Support Ratio']].rename(columns = {'Support Ratio' : 'nta'})

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
df_stay_group = df_stay[['country', 'year', 'yrs_stay']].groupby(['country', 'year']).mean().reset_index()
df_stay_group['yrs_stay_eff'] = 1 / (1 + np.exp(-(df_stay_group['yrs_stay'] * param_stay + 1)))
fig = px.line(df_stay_group, x = 'year', y = 'yrs_stay_eff', color = 'country')
fig.show()

df_prob = df_ag[fixed_vars + [x for x in df_ag.columns if '_fam' in x]]
df_prob = pd.melt(df_prob, id_vars=fixed_vars, value_vars=df_prob.columns[3:],
             value_name='fam_prob', var_name='year')
df_prob['year'] = df_prob['year'].apply(lambda x: int(x[:4]))
df_prob_group = df_prob[['country', 'year', 'fam_prob']].groupby(['country', 'year']).mean().reset_index()
df_prob_group['fam_prob_eff'] = 1 / (1 + np.exp(-(df_prob_group['fam_prob'] * param_fam + 10)))
fig = px.line(df_prob_group, x = 'year', y = 'fam_prob_eff', color = 'country')
fig.show()

df_ag_long = df_age.merge(df_stay, on = fixed_vars + ['year'], how = 'left').merge(df_prob, on = fixed_vars + ['year'], how = 'left')
df_ag_long = df_ag_long.merge(nta, on = 'age', how = 'left')

## remittances
df = pd.read_parquet("C:\\Data\\my_datasets\\italy\\simulation_data.parquet")
df['date'] = pd.to_datetime(df.date)
df.sort_values(['country', 'date'], inplace = True)
df_rem_group = df[~df[["date", "country"]].duplicated()][["date", "country", "remittances", "gdp_per_capita", "delta_gdp"]]
df_pop_group = df[["date", "country", "population"]].groupby(["date", "country"]).sum().reset_index()
df_rem_group = df_rem_group.merge(df_pop_group, on = ["date", "country"], how = 'left')
df_rem_group['exp_pop'] = df_rem_group['remittances'] / rem_amount
df_rem_group['pct_sending'] = df_rem_group['exp_pop'] / df_rem_group['population']
df_rem_group['year'] = df_rem_group["date"].dt.year

country_avg = df_rem_group[['country', 'pct_sending']].groupby('country').mean()
country_avg['pct_sending'].hist(bins = 50)
plt.show(block=True)

df_year_gdp = df_rem_group[['country', 'year', "gdp_per_capita", "delta_gdp"]].groupby(['country', 'year']).mean().reset_index()
df_ag_long = df_ag_long.merge(df_year_gdp, on = ['country', 'year'], how = 'left')

### normalise delta gdp
df_ag_long["delta_gdp_norm"] = df_ag_long.delta_gdp / abs(df_ag_long.delta_gdp.min())

gdp_group = df_ag_long[['country', 'year', 'delta_gdp_norm']].groupby(['country', 'year']).mean().reset_index()

fig = px.bar(gdp_group[gdp_group.year == 2015], x = 'country', y = 'delta_gdp_norm')
fig.show()

gdp_group['gdp_eff'] = 1 / (1 + np.exp(-(gdp_group['delta_gdp_norm'] * param_gdp)))
fig = px.line(gdp_group, x = 'year', y = 'gdp_eff', color = 'country')
fig.show()

#group nta
nta_group = df_ag_long[['country', 'year', 'nta']].groupby(['country', 'year']).mean().reset_index()

def show_cumulative_effects_components(country):
    df_nta_country = nta_group[nta_group.country == country][['year', 'country', 'nta']].copy()
    df_nta_country['nta'] = pd.to_numeric(df_nta_country['nta'])
    df_nta_country['nta_prob'] = 1 / (1 + np.exp(-(df_nta_country['nta'])))

    df_stay_country = df_stay_group[df_stay_group.country == country][['year', 'country', 'yrs_stay']].copy()
    df_stay_country = df_stay_country.merge(df_nta_country[['year', 'country', 'nta_prob', 'nta']], on = ['year', 'country'])
    df_stay_country['yrs_stay_eff'] = -1 * (df_stay_country['nta_prob'] - (1 / (1 + np.exp(-(
            df_stay_country['yrs_stay'] * param_stay + df_stay_country['nta'])))))

    df_fam_country = df_prob_group[df_prob_group.country == country][['year', 'country', 'fam_prob']].copy()
    df_fam_country = df_fam_country.merge(df_nta_country[['year', 'country', 'nta_prob', 'nta']], on = ['year', 'country'])
    df_fam_country['fam_eff'] = -1 * (df_fam_country['nta_prob'] - (1 / (1 + np.exp(-(
            df_fam_country['fam_prob'] * param_fam + df_fam_country['nta'])))))

    df_gdp_country = gdp_group[gdp_group.country == country][['year', 'country', 'delta_gdp_norm']].copy()
    df_gdp_country = df_gdp_country.merge(df_nta_country[['year', 'country', 'nta_prob', 'nta']], on = ['year', 'country'])
    df_gdp_country['gdp_eff'] = -1 * (df_gdp_country['nta_prob'] - (1 / (1 + np.exp(-(
            df_gdp_country['delta_gdp_norm'] * param_gdp + df_gdp_country['nta'])))))

    df_country = df_stay_country.merge(df_fam_country, on = ['country', 'year', 'nta_prob', 'nta']).merge(df_gdp_country, on = ['country', 'year', 'nta_prob', 'nta'])
    df_country = df_country.melt(id_vars=['year', 'country'],value_vars=['yrs_stay_eff', 'fam_eff', 'gdp_eff'],var_name='Effect Type',value_name='Effect Value')

    fig = px.line(df_country, x='year', y='Effect Value', color='Effect Type',title=f'{country}: Effect Trends Over the Years',
                  labels={'Effect Value': 'Effect Size', 'year': 'Year'},template='plotly_white', markers=True)
    fig.add_trace(go.Scatter(x=df['year'], y=[0] * len(df['year']), mode='lines', name='Zero Line',line=dict(color='black', dash='dash')))
    fig.show()

show_cumulative_effects_components("Bangladesh")
show_cumulative_effects_components("Germany")

###
#parameters
param_nta = 1
param_stay = -0.2
param_fam = -2.5
param_gdp = -2
rem_amount = 5000
eq_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
dr_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
fl_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
st_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
tot_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
dict_dis_par = dict(zip(['eq', 'dr', 'fl', 'st', 'tot'], [eq_par, dr_par, fl_par, st_par, tot_par]))

# df_ag_long['theta'] =  param_nta * df_ag_long['nta'] + param_stay * df_ag_long['yrs_stay'] + param_fam * df_ag_long['fam_prob']
# df_ag_long['theta'] = pd.to_numeric(df_ag_long['theta'], errors='coerce')
# df_ag_long['prob'] = 1 / (1 + np.exp(-df_ag_long['theta']))
#
def plot_prob(df):
    df['theta'].hist()
    plt.title("Theta distribution")
    plt.show(block = True)
    #
    # df['prob_no_nta'].hist()
    # plt.title("Probability without NTA distribution")
    # plt.show(block = True)
    # plt.plot(df['prob_no_nta'].dropna().sort_values().tolist())
    # plt.title("Probability without NTA distribution")
    # plt.show(block = True)
    df['prob'].hist()
    plt.title("Probability distribution")
    plt.show(block = True)

    plt.plot(df['prob'].sort_values().tolist())
    plt.title("Probability distribution")
    plt.show(block=True)

# plot_prob(df_ag_long)
##############################
## iterate over parameters space
########################
nta_space = [1, 10, 20, 30]
stay_space = [-1, -3, -6, -9]
fam_space = [-1, -3, -6, -9]
gdp_space = [-0.00005, -0.0001, -0.0005]

matplotlib.use('Agg')
df_sample = df_ag_long.sample(100_000).copy()
out = "C:\\git-projects\\csh_remittances\\italy\\plots\\probabilities\\philippines\\"
df_sample = df_ag_long[df_ag_long.country == 'Philippines'].copy()
for param_nta in tqdm(nta_space):
    for param_stay in stay_space:
        for param_fam in fam_space:
            for param_gdp in gdp_space:
                df_sample['theta'] = (param_nta * df_sample['nta'] + param_stay * df_sample['yrs_stay'] +
                                      param_fam * df_sample['fam_prob'] + param_gdp * df_sample["delta_gdp"])
                df_sample['theta'] = pd.to_numeric(df_sample['theta'], errors='coerce')
                df_sample['prob'] = 1 / (1 + np.exp(-df_sample['theta']))
                df_sample.loc[df_sample.nta == 0, 'prob'] = 0

                fig, ax = plt.subplots()
                plt.plot(df_sample['prob'].sort_values().tolist())
                fig.savefig(out + f"prob_line_nta_{param_nta}_stay_{param_stay}_fam_{param_fam}_gdp_{param_gdp}.png")
                plt.close()

param_nta = 15
param_stay = -1
param_fam = -1
param_gdp = -0.00001

###
####################
# simulations
####################
matplotlib.use('QtAgg')
def simulate_one_country_no_disasters(country, plot, disable_progress = False):
    rem_country = df_rem_group[(df_rem_group.country == country)].copy()

    df_country = df_ag_long[df_ag_long.country == country].copy()
    df_country['theta'] =  param_nta * df_country['nta'] + param_stay * df_country['yrs_stay'] + param_fam * df_country['fam_prob'] + param_gdp * df_country["delta_gdp_norm"]
    df_country['theta'] = pd.to_numeric(df_country['theta'], errors='coerce')
    df_country['prob'] = 1 / (1 + np.exp(-df_country['theta']))
    df_country.loc[df_country.nta == 0, 'prob'] = 0

    plot_prob(df_country)

    senders = []
    for date in tqdm(rem_country.date, disable=disable_progress):
        probs = df_country[(df_country.year == date.year) & (~df_country.prob.isna())]['prob'].tolist()
        senders.append(np.random.binomial(1, probs).sum())
    res = rem_country[['date', 'remittances']].copy()
    res['simulated_senders'] = senders

    if plot:
        plot_remittances_senders(res)
    return res

nep_res = simulate_one_country_no_disasters("Philippines", False)
nep_res['sim_remittances'] = nep_res.simulated_senders * rem_amount
goodness_of_fit_results(nep_res)

def compute_disasters_theta():
    df_dis = df[~df[["date", "country"]].duplicated()][["date", "country"] + [x for x in df.columns[9:]]]
    df_dis.rename(columns = {'eq' : 'eq_0', 'st' : 'st_0', 'fl' : 'fl_0', 'dr' : 'dr_0', 'tot' : 'tot_0'}, inplace = True)
    for disaster in ['eq', 'dr', 'fl', 'st', 'tot']:
        params = dict_dis_par[disaster]
        impact =  sum([4 * params[int(x)] * df_dis[f"{disaster}_{int(x)}"] for x in np.linspace(0, 12, 13)])
        df_dis[f"{disaster}_score"] = impact
    return df_dis
df_dis = compute_disasters_theta()

def simulate_one_country_with_disasters(country, plot, disable_progress = False):
    rem_country = df[(df.country == country) & (df.age_group == 'Less than 5 years') & (df.sex == 'male')].copy()

    df_country = df_ag_long[df_ag_long.country == country].copy()
    df_country['theta'] =  param_nta * df_country['nta'] + param_stay * df_country['yrs_stay'] + param_fam * df_country['fam_prob'] + param_gdp * df_country["delta_gdp_norm"]
    df_country['theta'] = pd.to_numeric(df_country['theta'], errors='coerce')
    df_country.loc[df_country.nta == 0, 'theta'] = -100

    # plot_prob(df_country)

    senders = []
    for date in tqdm(rem_country.date, disable=disable_progress):
        thetas = (df_country[(df_country.year == date.year) & (~df_country.theta.isna())]['theta'] +
                  df_dis[(df_dis.country == country) & (df_dis.date == date)]['tot_score'].item() * 0.05)
        probs = 1 / (1 + np.exp(-thetas.dropna()))
        senders.append(np.random.binomial(1, probs).sum())
    res = rem_country[['date', 'remittances']].copy()
    res['simulated_senders'] = senders

    if plot:
        plot_remittances_senders(res)
    return res

nep_res = simulate_one_country_with_disasters("Bangladesh", True)
nep_res['sim_remittances'] = nep_res.simulated_senders * rem_amount
goodness_of_fit_results(nep_res)

def pct_senders_country(country, disasters = True, plot = False, disable_progress = False):
    if disasters:
        nep_res = simulate_one_country_with_disasters(country, plot = False, disable_progress = disable_progress)
    else:
        nep_res = simulate_one_country_no_disasters(country, plot=False, disable_progress = disable_progress)
    nep_res = nep_res.merge(df_rem_group[df_rem_group.country == country][['date', 'population', 'exp_pop', 'pct_sending']], on = 'date')
    nep_res['simulated_pct_senders'] = nep_res['simulated_senders'] / nep_res['population']
    nep_res['country'] = country

    if plot:
        plt.plot(nep_res['simulated_pct_senders'], label = 'simulated senders pct')
        plt.plot(nep_res['pct_sending'], label='real senders pct')
        plt.legend()
        plt.show(block = True)
    return nep_res[['date', 'country', 'simulated_pct_senders', 'pct_sending']]

nep_res = pct_senders_country("Bangladesh", plot = True)

large_senders = ['Bangladesh', 'China', 'India', 'Pakistan', 'Philippines', 'Sri Lanka']  #df_rem_group[df_rem_group.remittances > 20_000_000]['country'].unique().tolist()
df_res_pct = pd.DataFrame([])
for country in tqdm(large_senders):
    nep_res = pct_senders_country(country, disasters=True, plot=False, disable_progress = True)
    df_res_pct = pd.concat([df_res_pct, nep_res])

df_melted = df_res_pct.melt(id_vars=['date', 'country'],value_vars=['simulated_pct_senders', 'pct_sending'],
                             var_name='Type', value_name='Percentage')

fig = px.line(df_melted, x='date', y='Percentage', color='Type',
              facet_col='country', facet_col_wrap=3,  # Adjust for multiple countries
              title='Simulated vs. Actual Percentage of Senders',
              labels={'Percentage': 'Percentage Sending', 'date': 'Date'},
              template='plotly_white')

fig.update_layout(xaxis_tickangle=-45)
fig.show()

def simulate_all_countries(disasters = False):
    res = pd.DataFrame([])
    if disasters:
        for country in tqdm(df_ag_long.country.unique()):
            country_res = simulate_one_country_with_disasters(country, plot = False , disable_progress = True)
            country_res['country'] = country
            res = pd.concat([res, country_res])
    else:
        for country in tqdm(df_ag_long.country.unique()):
            country_res = simulate_one_country_no_disasters(country, plot=False, disable_progress=True)
            country_res['country'] = country
            res = pd.concat([res, country_res])
    return res

df_res = simulate_all_countries(disasters=True)
df_res['sim_remittances'] = df_res.simulated_senders * 3000
df_res_small = df_res[df_res['remittances'] > 10_000]
df_res_small = df_res_small[df_res_small['country'] != 'China']

plot_all_results_log(df_res_small)
plot_all_results_lin(df_res_small)
goodness_of_fit_results(df_res_small)

plot_correlation_senders_remittances(df_res_small)
plot_correlation_remittances(df_res_small)

