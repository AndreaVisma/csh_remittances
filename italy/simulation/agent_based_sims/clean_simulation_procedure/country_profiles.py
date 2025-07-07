
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mtick
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = 'browser'
pd.options.mode.chained_assignment = None

## import from other scripts
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import random
from italy.simulation.agent_based_sims.clean_simulation_procedure.load_all_data import load_data_and_param, compute_disasters_theta
from italy.simulation.agent_based_sims.clean_simulation_procedure.functions_for_simulation import *
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
param_albania  = 0 #was -2.2
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
            df_country = df_ag_long[(df_ag_long.country == country) & (df_ag_long.age == 40) & (df_ag_long.yrs_stay == 5)].copy()
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
                          df_dis[(df_dis.country == country) & (df_dis.date == date)]['tot_score'].item() * 0.1)
            df_country = df_country[~df_country.theta.isna()][['country', 'theta']]
            df_all_senders = pd.concat([df_all_senders, df_country])
        else:
            df_country = df_ag_long[
                (df_ag_long.country == country) & (df_ag_long.age == 40) & (df_ag_long.yrs_stay == 5)].copy()
            df_country = df_country[(~df_country.year.duplicated())]
            df_country['theta'] = (
                    param_nta * df_country['nta'] + param_stay * df_country['yrs_stay'] + param_fam * df_country[
                'fam_prob']
                    + param_gdp * df_country["delta_gdp_norm"] +
                    param_close * df_country["close_country"] + param_albania * df_country["albania"])
            df_country['theta'] = pd.to_numeric(df_country['theta'], errors='coerce')
            df_country.loc[df_country.nta == 0, 'theta'] = -100
            df_country = df_country[~df_country.theta.isna()][['country', 'theta']]
            df_all_senders = pd.concat([df_all_senders, df_country])

    return df_all_senders

df_all_senders = return_representative_agent_profile()

def plot_representative_agents_profile(disasters = True, disable_progress = True):
    df_all_senders = return_representative_agent_profile(disasters=disasters)
    df_mean_sender = df_all_senders.groupby('country').mean().reset_index()
    df_mean_sender['prob'] = (1 / (1 + np.exp(-df_mean_sender['theta'])))

    fig = px.scatter(df_mean_sender, 'theta', 'prob', color = 'country')
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(title = "Probability for a representative 40 years old person")
    fig.update_yaxes(title = "Probability")
    fig.update_xaxes(title="Theta")
    fig.show()
    return df_all_senders

plot_representative_agents_profile(disasters=False)

def return_population_profile(countries, years, disasters = True, disable_progress = True):
    df_all_senders = pd.DataFrame([])
    for country in tqdm(countries):
        if disasters:
            rem_country = df_rem_group[(df_rem_group.country == country) & (df_rem_group.year.isin(years))].copy()

            df_country = df_ag_long[(df_ag_long.country == country) & (df_ag_long.year.isin(years))].copy()
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
                          df_dis[(df_dis.country == country) & (df_dis.date == date)]['tot_score'].item() * 0.1)
                df_country.loc[(df_country.year == date.year) & (~df_country.theta.isna()), 'prob'] = (
                        1 / (1 + np.exp(-df_country.loc[(df_country.year == date.year) & (~df_country.theta.isna()), 'theta'])))
            df_country.loc[df_country.nta == 0, 'prob'] = 0
        else:
            nep_res = simulate_one_country_no_disasters(country, plot=False, disable_progress = disable_progress)
        df_country = df_country[~df_country.prob.isna()][['country', 'prob']]
        df_all_senders = pd.concat([df_all_senders, df_country])
    return df_all_senders

def plot_population_profile(countries, years):

    df_all_senders = return_population_profile(countries, years)

    fig, ax = plt.subplots(figsize=(6, 6))
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
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.grid()
    plt.savefig("C:\\Users\\Andrea Vismara\\Desktop\\papers\\remittances\\charts\\profile.svg")
    plt.show(block = True)


plot_population_profile(countries = ['Bangladesh', 'Peru', 'Romania', 'Germany'], years = [2015, 2016, 2017, 2018, 2019,2020, 2021, 2022])

def disasters_remittances_one_country(country, plot = False, disable_progress = False):

    res_no = simulate_one_country_no_disasters(country, dem_params = [param_nta, param_stay, param_fam, param_gdp, param_close, param_albania],
                                               df_rem_group = df_rem_group, df_ag_long=df_ag_long, plot = False, disable_progress = disable_progress)
    res_with = simulate_one_country_with_disasters(df_dis, df, country, plot = False)
    res = res_no.merge(res_with[['date', 'simulated_senders']], on = 'date', suffixes = ('_no', '_with'))
    res['remittances_sim_no'] = res['simulated_senders_no'] * rem_amount
    res['remittances_sim_with'] = res['simulated_senders_with'] * rem_amount

    # Calculate differences
    res["difference"] = res["remittances_sim_with"] - res["remittances_sim_no"]
    # res["cumulative_remittances_no"] = res["remittances_sim_no"].cumsum()
    res["cumulative_remittances_with"] = res["remittances_sim_with"].cumsum()
    res["cumulative_difference"] = res["difference"].cumsum()

    if plot:
        # Create the plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=res["date"], y=res["remittances_sim_no"], mode="lines+markers", name="Simulation no disasters"))
        fig.add_trace(go.Scatter(x=res["date"], y=res["remittances_sim_with"], mode="lines+markers", name="Simulation with disasters"))
        fig.add_trace(go.Bar(x=res["date"], y=res["difference"], name="Difference"))

        fig.add_trace(go.Scatter(x=res["date"], y=res["cumulative_difference"], mode="lines+markers", name="Cumulative difference (rhs)",
                                 yaxis="y2"))

        fig.update_layout(
            title=f"Remittances Simulation Comparison, {country}",
            xaxis_title="Date",
            yaxis_title="Remittances",
            yaxis2=dict(title="Cumulative Difference", overlaying="y", side="right"),
            barmode="group",
            template="plotly_white"
        )

        fig.show()
    return res

res = disasters_remittances_one_country('Bangladesh', plot = True, disable_progress=True)

all_res = pd.DataFrame([])
for country in tqdm(df_ag_long.country.unique()):
    res = disasters_remittances_one_country(country, plot = False, disable_progress=True)
    res['country'] = country
    all_res = pd.concat([all_res, res])
all_res = all_res.merge(df_rem_group[['date', 'country', 'remittances']], on = ['date', 'country'])

all_res_last = all_res[all_res.columns[1:]].groupby('country').sum().reset_index()
all_res_last['pct_because_disasters'] = 100 * all_res_last['difference'] / all_res_last['remittances_x']

#####
fig = px.scatter(all_res_last, x = 'remittances_x', y = "pct_because_disasters", color = 'country', log_x=True)
fig.show()
####

import statsmodels.api as sm
X = all_res_last["remittances_x"].to_numpy()
X = sm.add_constant(X)
### no disasters
y = all_res_last["remittances_sim_no"].to_numpy()
results = sm.OLS(y, X).fit()
all_res_last["sim_remittances_no_line"] = results.predict(X)
### with disasters
y = all_res_last["remittances_sim_with"].to_numpy()
results = sm.OLS(y, X).fit()
all_res_last["sim_remittances_with_line"] = results.predict(X)


fig = px.scatter(all_res_last, x = 'remittances_x', y = "remittances_sim_with", color = 'country')
fig.add_trace(
    go.Scatter(x=all_res_last["remittances_x"], y=all_res_last["sim_remittances_with_line"], mode="lines", name="Trendline with disasters"))
fig.add_trace(
    go.Scatter(x=all_res_last["remittances_x"], y=all_res_last["remittances_x"], mode="lines", name="Perfect match"))
fig.show()

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=all_res_last["remittances_x"], y=all_res_last["remittances_sim_no"], mode="markers", name="Simulation no disasters"))
fig.add_trace(
    go.Scatter(x=all_res_last["remittances_x"], y=all_res_last["sim_remittances_no_line"], mode="lines", name="Trendline no disasters"))
fig.add_trace(
    go.Scatter(x=all_res_last["remittances_x"], y=all_res_last["remittances_sim_with"], mode="markers", name="Simulation with disasters"))
fig.add_trace(
    go.Scatter(x=all_res_last["remittances_x"], y=all_res_last["sim_remittances_with_line"], mode="lines", name="Trendline with disasters"))

fig.show()

print(f"""Total remittances sent over the 2008-2022 period:
{round((all_res_last["remittances_x"].sum()) / 1_000_000_000, 2)} billion euros""")
print(f"""Additional remittances sent because of disasters:
{round((all_res_last["remittances_sim_with"].sum() - all_res_last["remittances_sim_no"].sum()) / 1_000_000_000, 2)} billion euros""")
print(f"""These represent the {round(100 * (all_res_last["remittances_sim_with"].sum() - all_res_last["remittances_sim_no"].sum()) / all_res_last["remittances_x"].sum(), 2)}% of all remittances""")


