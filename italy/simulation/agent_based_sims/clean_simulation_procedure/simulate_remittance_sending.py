
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
from italy.simulation.agent_based_sims.clean_simulation_procedure.load_all_data import load_data_and_param, compute_disasters_theta

from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)

#######
# load all data!
#######
(params, df_ag_long, df_rem_group, df_stay_group,
 df_prob_group, gdp_group, nta_group, df) = load_data_and_param(fixed_vars =['agent_id', 'country', 'sex'])

### Define params
param_nta, param_stay, param_fam, param_gdp, rem_amount = params
param_close = -1.5
param_albania  = -2.2
eq_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
dr_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
fl_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
st_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
tot_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
dict_dis_par = dict(zip(['eq', 'dr', 'fl', 'st', 'tot'], [eq_par, dr_par, fl_par, st_par, tot_par]))

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

matplotlib.use('QtAgg')
def simulate_one_country_no_disasters(country, plot, disable_progress = False):
    rem_country = df_rem_group[(df_rem_group.country == country)].copy()

    df_country = df_ag_long[df_ag_long.country == country].copy()
    df_country['theta'] =  (param_nta * df_country['nta'] + param_stay * df_country['yrs_stay'] + param_fam * df_country['fam_prob']
                            + param_gdp * df_country["delta_gdp_norm"] + param_close * df_country["close_country"]  + param_albania * df_country["albania"])
    df_country['theta'] = pd.to_numeric(df_country['theta'], errors='coerce')
    df_country['prob'] = 1 / (1 + np.exp(-df_country['theta']))
    df_country.loc[df_country.nta == 0, 'prob'] = 0

    # plot_prob(df_country)

    senders = []
    for date in tqdm(rem_country.date, disable=disable_progress):
        probs = df_country[(df_country.year == date.year) & (~df_country.prob.isna())]['prob'].tolist()
        senders.append(np.random.binomial(1, probs).sum())
    res = rem_country[['date', 'remittances']].copy()
    res['simulated_senders'] = senders

    if plot:
        plot_remittances_senders(res)
    return res

nep_res = simulate_one_country_no_disasters("Bangladesh", True)
nep_res['sim_remittances'] = nep_res.simulated_senders * rem_amount
goodness_of_fit_results(nep_res)

df_dis = compute_disasters_theta(df, dict_dis_par)
def simulate_one_country_with_disasters(df_dis, country, plot, reprocess_dis = False, disable_progress = False):

    if reprocess_dis:
        df_dis = compute_disasters_theta(df, dict_dis_par)
    rem_country = df[(df.country == country) & (df.age_group == 'Less than 5 years') & (df.sex == 'male')].copy()

    df_country = df_ag_long[df_ag_long.country == country].copy()
    df_country['theta'] =  (param_nta * df_country['nta'] + param_stay * df_country['yrs_stay'] + param_fam * df_country['fam_prob']
                            + param_gdp * df_country["delta_gdp_norm"] +
                            param_close * df_country["close_country"] + param_albania * df_country["albania"])
    df_country['theta'] = pd.to_numeric(df_country['theta'], errors='coerce')
    df_country.loc[df_country.nta == 0, 'theta'] = -100

    # plot_prob(df_country)

    senders = []
    for date in tqdm(rem_country.date, disable=disable_progress):
        thetas = (df_country[(df_country.year == date.year) & (~df_country.theta.isna())]['theta'] +
                  df_dis[(df_dis.country == country) & (df_dis.date == date)]['tot_score'].item() * 0.3)
        probs = 1 / (1 + np.exp(-thetas.dropna()))
        senders.append(np.random.binomial(1, probs).sum())
    res = rem_country[['date', 'remittances']].copy()
    res['simulated_senders'] = senders

    if plot:
        plot_remittances_senders(res)
    return res

nep_res = simulate_one_country_with_disasters(df_dis, "Bangladesh", True)
nep_res['sim_remittances'] = nep_res.simulated_senders * rem_amount
goodness_of_fit_results(nep_res)

def simulate_all_countries(disasters = False):
    res = pd.DataFrame([])
    if disasters:
        for country in tqdm(df_ag_long.country.unique()):
            country_res = simulate_one_country_with_disasters(df_dis, country, plot = False , disable_progress = True)
            country_res['country'] = country
            res = pd.concat([res, country_res])
    else:
        for country in tqdm(df_ag_long.country.unique()):
            country_res = simulate_one_country_no_disasters(country, plot=False, disable_progress=True)
            country_res['country'] = country
            res = pd.concat([res, country_res])
    return res

def plot_everything_about_results(df):
    plot_all_results_log(df)
    plot_all_results_lin(df)
    goodness_of_fit_results(df)

    plot_correlation_senders_remittances(df)
    plot_correlation_remittances(df)

def plot_country_mean(df):
    df_mean = df[['country', 'remittances', 'sim_remittances']].groupby(['country']).mean().reset_index()
    fig = px.scatter(df_mean, x = 'remittances', y = 'sim_remittances',
                     color = 'country', log_x=True, log_y=True)
    fig.add_scatter(x=np.linspace(0, df_mean.remittances.max(), 100),
                    y=np.linspace(0, df_mean.remittances.max(), 100))
    fig.show()
    goodness_of_fit_results(df_mean)

df_res = simulate_all_countries(disasters=True)
df_res['sim_remittances'] = df_res.simulated_senders * rem_amount
plot_everything_about_results(df_res)
plot_country_mean(df_res)

df_res_small = df_res[df_res['remittances'] > 10_000]
df_res_small = df_res_small[df_res_small['country'] != 'China']
plot_everything_about_results(df_res_small)
plot_country_mean(df_res_small)

## plot how close the real percentage of senders and the simulated one are over time
def pct_senders_country(country, disasters = True, plot = False, disable_progress = False):
    if disasters:
        nep_res = simulate_one_country_with_disasters(df_dis, country, plot = False, disable_progress = disable_progress)
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

nep_res = pct_senders_country("Philippines", plot = True)
