

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
from scipy.interpolate import interp1d
from italy.simulation.agent_based_sims.clean_simulation_procedure.load_all_data import load_data_and_param, compute_disasters_theta

from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)
from italy.simulation.agent_based_sims.clean_simulation_procedure.simulate_remittance_sending import simulate_one_country_with_disasters, simulate_one_country_no_disasters

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

df_dis = compute_disasters_theta(df, dict_dis_par)
#########
# define population profiles
#########

def return_representative_agent_profile(disasters = True, disable_progress = True):
    df_all_senders = pd.DataFrame([])

    for country in tqdm(df.country.unique()):
        if disasters:
            rem_country = df_rem_group[(df_rem_group.country == country)].copy()
            df_country = df_ag_long[(df_ag_long.country == country) & (df_ag_long.age == 40)].copy()
            df_country = df_country[(~df_country.year.duplicated())]
            df_country['theta'] = (
                        param_nta * df_country['nta'] + param_stay * df_country['yrs_stay'] + param_fam * df_country[
                    'fam_prob']
                        + param_gdp * df_country["delta_gdp_norm"] +
                        param_close * df_country["close_country"] + param_albania * df_country["albania"])
            df_country['theta'] = pd.to_numeric(df_country['theta'], errors='coerce')
            df_country.loc[df_country.nta == 0, 'theta'] = -100

            for date in tqdm(rem_country.date, disable=disable_progress):
                df_country.loc[(df_country.year == date.year) & (~df_country.theta.isna()), 'theta'] = \
                    (df_country[(df_country.year == date.year) & (~df_country.theta.isna())]['theta'] +
                          df_dis[(df_dis.country == country) & (df_dis.date == date)]['tot_score'].item() * 0.3)
                df_country.loc[(df_country.year == date.year) & (~df_country.theta.isna()), 'prob'] = (
                        1 / (1 + np.exp(-df_country.loc[(df_country.year == date.year) & (~df_country.theta.isna()), 'theta'])))
            df_country.loc[df_country.nta == 0, 'prob'] = 0
            df_country = df_country[~df_country.prob.isna()][['country', 'prob', 'theta']]
            df_all_senders = pd.concat([df_all_senders, df_country])
    else:
        nep_res = simulate_one_country_no_disasters(country, plot=False, disable_progress=disable_progress)

    return df_all_senders

df_all_senders = return_representative_agent_profile()

def plot_representative_agents_profile(disasters = True, disable_progress = True):
    df_all_senders = return_representative_agent_profile()
    df_mean_sender = df_all_senders.groupby('country').mean().reset_index()

    # plt.scatter(df_all_senders['theta'], df_all_senders['prob'])
    #
    # plt.xlabel("Total Diaspora population")
    # plt.ylabel("Probability")
    # plt.title("Probability Profiles for Different Countries")
    # plt.legend()
    # plt.show(block = True)
    #
    # plt.scatter(df_mean_sender['theta'], df_mean_sender['prob'])
    #
    # plt.xlabel("Total Diaspora population")
    # plt.ylabel("Probability")
    # plt.title("Probability Profiles for Different Countries")
    # plt.legend()
    # plt.show(block = True)

    fig = px.scatter(df_mean_sender, 'theta', 'prob', color = 'country')
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(title = "Probability for a representative 40 years old person")
    fig.update_yaxes(title = "Probability")
    fig.update_xaxes(title="Theta")
    fig.show()


def return_population_profile(countries, disasters = True, disable_progress = True):
    df_all_senders = pd.DataFrame([])
    for country in tqdm(countries):
        if disasters:
            rem_country = df_rem_group[(df_rem_group.country == country)].copy()

            df_country = df_ag_long[df_ag_long.country == country].copy()
            df_country['theta'] = (
                        param_nta * df_country['nta'] + param_stay * df_country['yrs_stay'] + param_fam * df_country[
                    'fam_prob']
                        + param_gdp * df_country["delta_gdp_norm"] +
                        param_close * df_country["close_country"] + param_albania * df_country["albania"])
            df_country['theta'] = pd.to_numeric(df_country['theta'], errors='coerce')
            df_country.loc[df_country.nta == 0, 'theta'] = -100

            for date in tqdm(rem_country.date, disable=disable_progress):
                df_country.loc[(df_country.year == date.year) & (~df_country.theta.isna()), 'theta'] = \
                    (df_country[(df_country.year == date.year) & (~df_country.theta.isna())]['theta'] +
                          df_dis[(df_dis.country == country) & (df_dis.date == date)]['tot_score'].item() * 0.3)
                df_country.loc[(df_country.year == date.year) & (~df_country.theta.isna()), 'prob'] = (
                        1 / (1 + np.exp(-df_country.loc[(df_country.year == date.year) & (~df_country.theta.isna()), 'theta'])))
            df_country.loc[df_country.nta == 0, 'prob'] = 0
        else:
            nep_res = simulate_one_country_no_disasters(country, plot=False, disable_progress = disable_progress)
        df_country = df_country[~df_country.prob.isna()][['country', 'prob']]
        df_all_senders = pd.concat([df_all_senders, df_country])
    return df_all_senders

def plot_population_profile(countries):

    df_all_senders = return_population_profile(countries)

    plt.figure(figsize=(10, 6))
    x_common = np.linspace(0, 1, 100)  # Common x-axis (normalized scale)

    for country in tqdm(countries):
        probs = np.sort(df_all_senders[df_all_senders["country"] == country]["prob"].values)
        x_original = np.linspace(0, 1, len(probs))  # Original scale (variable length)
        f = interp1d(x_original, probs, kind='linear', fill_value="extrapolate")
        probs_interp = f(x_common)
        plt.plot(x_common, probs_interp, label=country, lw=3)

    plt.xlabel("Total Diaspora population")
    plt.ylabel("Probability")
    plt.title("Probability Profiles for Different Countries")
    plt.legend()
    plt.show(block = True)

plot_population_profile(countries = ['Bangladesh', 'Germany', 'El Salvador', 'Mexico', 'Romania'])