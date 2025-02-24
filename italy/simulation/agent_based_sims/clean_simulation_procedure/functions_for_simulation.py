
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

def simulate_one_country_no_disasters(country, dem_params, df_rem_group, df_ag_long, plot, disable_progress = False):
    param_nta, param_stay, param_fam, param_gdp = dem_params
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

def simulate_all_countries(dem_params, df_rem_group, df_ag_long, disasters = False):
    res = pd.DataFrame([])
    if disasters:
        for country in tqdm(df_ag_long.country.unique()):
            country_res = simulate_one_country_with_disasters(df_dis, country, plot = False , disable_progress = True)
            country_res['country'] = country
            res = pd.concat([res, country_res])
    else:
        for country in tqdm(df_ag_long.country.unique()):
            country_res = simulate_one_country_no_disasters(country, dem_params = dem_params, df_rem_group = df_rem_group,
                                                            df_ag_long = df_ag_long, plot=False, disable_progress=True)
            country_res['country'] = country
            res = pd.concat([res, country_res])
    return res

def simulate_all_countries_deterministic_no_dis(dem_params, df_rem_group, df_ag_long):
    param_nta, param_stay, param_fam, param_gdp = dem_params

    df_ag_long['theta'] = (
                param_nta * df_ag_long['nta'] + param_stay * df_ag_long['yrs_stay'] + param_fam * df_ag_long['fam_prob']
                + param_gdp * df_ag_long["delta_gdp_norm"] + param_close * df_ag_long["close_country"] + param_albania *
                df_ag_long["albania"])
    df_ag_long['theta'] = pd.to_numeric(df_ag_long['theta'], errors='coerce')
    df_ag_long['prob'] = 1 / (1 + np.exp(-df_ag_long['theta']))
    df_ag_long.loc[df_ag_long.nta == 0, 'prob'] = 0
    df_ag_group = df_ag_long[['country', 'year', 'prob']].groupby(['country', 'year']).sum().reset_index()
    df_ag_group.rename(columns = {'prob' : 'simulated_senders'}, inplace = True)

    res = df_rem_group[['date', 'year', 'country', 'remittances']].copy()
    res = res.merge(df_ag_group, on = ['country', 'year'])

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