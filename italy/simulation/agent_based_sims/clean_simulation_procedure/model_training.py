
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
pd.options.mode.chained_assignment = None

## import from other scripts
from scipy.optimize import minimize
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

##############################
# calibrate simulation based only on demographic parameters
##############################

def loss_demographic_effect(X):
    df_res = simulate_all_countries_deterministic_no_dis(dem_params = X, df_rem_group = df_rem_sample, df_ag_long = df_long_sample)
    df_res['sim_remittances'] = df_res.simulated_senders * rem_amount
    df_res['error'] = df_res['remittances'] - df_res['sim_remittances']
    return round(np.mean(np.square(df_res['error'])) / 100_000_000_000, 3)

# 50% sample training
random.seed(1234)
sample_countries = random.sample(df_rem_group.country.unique().tolist(), int(0.4 * len(df_rem_group.country.unique())))
df_long_sample = df_ag_long[df_ag_long.country.isin(sample_countries)]
df_rem_sample = df_rem_group[df_rem_group.country.isin(sample_countries)]

# define initial parameters
X = [param_nta, param_stay, param_fam, param_gdp, param_close, param_albania]
error_basic = loss_demographic_effect(X)
print(error_basic)

## define function to search space
def coordinate_descent_demographic(initial_params, bounds, step_size=0.1, max_iter=20):
    current_params = initial_params.copy()
    best_mse = loss_demographic_effect(current_params)
    print(f"Current best MSE: {best_mse}")

    for _ in tqdm(range(max_iter)):
        for i in range(len(current_params) - 1): #perturbates all parameters apart from the albania param
            # Perturb parameter i
            param = current_params[i]
            lower, upper = bounds[i]
            delta = step_size * (upper - lower)

            # Test positive perturbation
            params_plus = current_params.copy()
            params_plus[i] = min(param + delta, upper)
            mse_plus = loss_demographic_effect(params_plus)

            # Test negative perturbation
            params_minus = current_params.copy()
            params_minus[i] = max(param - delta, lower)
            mse_minus = loss_demographic_effect(params_minus)

            # Update parameter in the best direction
            if mse_plus < best_mse:
                current_params = params_plus
                best_mse = mse_plus
                print(f"New best MSE: {best_mse}")
            elif mse_minus < best_mse:
                current_params = params_minus
                best_mse = mse_minus
                print(f"New best MSE: {best_mse}")
    return current_params, best_mse

### calibrate demographic parameters on training sample
initial_params = [param_nta, param_stay, param_fam, param_gdp, param_close, 0]
bounds = [[0.5, 1.5], [-1, 0], [-4, 0], [-4, 0], [-2.5, 0]]
current_params, best_mse = coordinate_descent(initial_params = initial_params, bounds = bounds, max_iter=8)

df_res = simulate_all_countries_deterministic_no_dis(dem_params=current_params, df_rem_group=df_rem_sample, df_ag_long=df_long_sample)
df_res['sim_remittances'] = df_res.simulated_senders * rem_amount
plot_everything_about_results(df_res, pred=False)
plot_country_mean(df_res)

## plot results for test sample
df_long_sample = df_ag_long[~df_ag_long.country.isin(sample_countries)]
df_long_sample = df_long_sample[df_long_sample.nta > 0]
df_rem_sample = df_rem_group[~df_rem_group.country.isin(sample_countries)]
df_res = simulate_all_countries_deterministic_no_dis(dem_params=current_params, df_rem_group=df_rem_sample, df_ag_long=df_long_sample)
df_res['sim_remittances'] = df_res.simulated_senders * rem_amount
plot_everything_about_results(df_res, pred=True)
plot_country_mean(df_res)

###########################
# calibrate disasters impact
###########################
# param_nta, param_stay, param_fam, param_gdp, param_close, param_albania = [1.1, -0.1, 0, -1.6, -2.5, 0]
param_nta, param_stay, param_fam, param_gdp, param_close, param_albania = [1, -0.1, -2, -2, -1.5, 0]

param_groups = ['eq', 'dr', 'fl', 'st', 'tot']
param_vector = [p for group in param_groups for p in dict_dis_par[group]]
bounds = [(-0.5, 1.0)] * len(param_vector)

def loss_disasters(X):
    df_res = simulate_all_countries_deterministic_with_dis(dem_params = [param_nta, param_stay, param_fam, param_gdp, param_close, param_albania],
                                                             df_rem_group = df_rem_sample,
                                                             df_ag_long = df_long_sample, disaster_vector = X)
    df_res['sim_remittances'] = df_res.simulated_senders * rem_amount
    df_res['error'] = df_res['remittances'] - df_res['sim_remittances']
    return round(np.mean(np.square(df_res['error'])) / 100_000_000_000, 3)

def coordinate_descent_tot_disasters(dict_dis_par, bounds, disaster = 'tot', step_size=0.1, max_iter=5):
    current_params = [p for group in param_groups for p in dict_dis_par[group]]
    best_mse = loss_disasters(current_params)
    print(f"Current best MSE: {best_mse}")

    params_to_vary = dict_dis_par[disaster]
    copy_dict_dis_plus = dict_dis_par.copy()
    copy_dict_dis_minus = dict_dis_par.copy()
    for _ in tqdm(range(max_iter)):
        for i in range(len(params_to_vary)):  # perturbates all parameters apart from the albania param
            # Perturb parameter i
            param = current_params[i]
            lower, upper = bounds[i]
            delta = step_size * (upper - lower)

            # Test positive perturbation
            params_plus = current_params.copy()
            params_plus[i] = min(param + delta, upper)
            copy_dict_dis_plus[disaster] = params_plus
            params_plus = [p for group in param_groups for p in copy_dict_dis_plus[group]]
            mse_plus = loss_disasters(params_plus)

            # Test negative perturbation
            params_minus = current_params.copy()
            params_minus[i] = max(param - delta, lower)
            copy_dict_dis_minus[disaster] = params_minus
            params_minus = [p for group in param_groups for p in copy_dict_dis_minus[group]]
            mse_minus = loss_disasters(params_minus)

            # Update parameter in the best direction
            if mse_plus < best_mse:
                current_params = params_plus
                best_mse = mse_plus
                print(f"New best MSE: {best_mse}")
            elif mse_minus < best_mse:
                current_params = params_minus
                best_mse = mse_minus
                print(f"New best MSE: {best_mse}")
    return current_params, best_mse

optimized_params, final_mse = coordinate_descent_tot_disasters(
    dict_dis_par, bounds, step_size=0.1, max_iter=5
)

##### do some tries
param_groups = ['eq', 'dr', 'fl', 'st', 'tot']
param_vector = [p for group in param_groups for p in dict_dis_par[group]]
df_res = simulate_all_countries_deterministic_with_dis(
    dem_params=[param_nta, param_stay, param_fam, param_gdp, param_close, param_albania],
    df_rem_group=df_rem_sample,
    df_ag_long=df_long_sample, disaster_vector=param_vector)
df_res['sim_remittances'] = df_res.simulated_senders * rem_amount
df_res['error'] = df_res['remittances'] - df_res['sim_remittances']
print(round(np.mean(np.square(df_res['error'])) / 100_000_000_000, 3))
plot_everything_about_results(df_res, pred=False)
plot_country_mean(df_res)














##################################
##################################
def loss_remittances_amount(rem_amount):
    df_res['sim_remittances'] = df_res.simulated_senders * rem_amount
    df_res['error'] = df_res['remittances'] - df_res['sim_remittances']
    return np.mean(np.square(df_res['error']))

def minimize_k():
    # df_res = simulate_all_countries(disasters=False)
    initial_guess = 3000
    res = minimize(loss_remittances_amount, initial_guess)
    print(f"Best value of remittances amount for current probability prediction: {round(res.x[0], 2)}")
    return round(res.x[0], 2)

rem_amount = minimize_k()
df_res['sim_remittances'] = df_res.simulated_senders * rem_amount

df_res_small = df_res[df_res['remittances'] > 10_000]
plot_everything_about_results(df_res_small)
plot_country_mean(df_res_small)


######
# split in training and test sample
######
# 50% sample training
random.seed(1234)
sample_countries = random.sample(df_rem_group.country.unique().tolist(), int(0.8 * len(df_rem_group.country.unique())))
df_long_sample = df_ag_long[df_ag_long.country.isin(sample_countries)]
df_rem_sample = df_rem_group[df_rem_group.country.isin(sample_countries)]

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

df_res = simulate_all_countries_deterministic_no_dis(dem_params = X, df_rem_group = df_rem_group,
                                                            df_ag_long = df_ag_long)
df_res['sim_remittances'] = df_res.simulated_senders * rem_amount
goodness_of_fit_results(df_res)

def loss_demographic_effect(X):
    df_res = simulate_all_countries_deterministic_no_dis(dem_params = X, df_rem_group = df_rem_sample, df_ag_long = df_long_sample)
    df_res['sim_remittances'] = df_res.simulated_senders * rem_amount
    df_res['error'] = df_res['remittances'] - df_res['sim_remittances']
    return np.mean(np.square(df_res['error']))

def minimize_params():
    # df_res = simulate_all_countries(disasters=False)
    initial_guess = [1, -0.1, -2, -2]
    res = minimize(loss_demographic_effect, initial_guess, method='L-BFGS-B')
    # print(f"Best value of remittances amount for current probability prediction: {round(res.x[0], 2)}")
    return res

res = minimize_params()