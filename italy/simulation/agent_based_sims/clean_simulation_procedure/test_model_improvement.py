

import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.tools.eval_measures import aic, bic
from scipy.stats import chi2
from sklearn.metrics import log_loss
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

df_dis = compute_disasters_theta(df, dict_dis_par)
dem_params = [param_nta, param_stay, param_fam, param_gdp, param_close, param_albania]

############
# simulate results from both models
############
results_no_dis = simulate_all_countries(df_dis, df, dem_params, df_rem_group, df_ag_long, disasters = False)
results_with_dis = simulate_all_countries(df_dis, df, dem_params, df_rem_group, df_ag_long, disasters = True)

results_no_dis['sim_remittances'] = results_no_dis['simulated_senders'] * rem_amount
results_with_dis['sim_remittances'] = results_with_dis['simulated_senders'] * rem_amount

results_no_dis['error'] = results_no_dis['remittances'] - results_no_dis['sim_remittances']
results_with_dis['error'] = results_with_dis['remittances'] - results_with_dis['sim_remittances']

#######
# Use a paired t-test to compare forecast errors.
#######
errors_no_dis = results_no_dis['error'].to_numpy()  # Errors from first model
errors_with_dis = results_with_dis['error'].to_numpy()  # Errors from second model

# Paired t-test
t_stat, p_value = ttest_rel(errors_no_dis, errors_with_dis)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

if p_value < 0.05:
    print("The model with disasters has significantly better performance.")
else:
    print("No significant improvement; the model without disasters might be preferable.")

#######
# Use Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) to balance improvement vs. complexity.
#######

y = results_no_dis['remittances'].to_numpy()

# Log-likelihood no disasters
y_pred = results_no_dis['sim_remittances'].to_numpy()
residuals = y - y_pred
num_params_1 = 6
sigma2 = np.var(residuals, ddof=num_params_1)
n = len(y)
log_likelihood_no = -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(sigma2) - np.sum(residuals**2) / (2 * sigma2)
print(f"Log-Likelihood without disasters: {log_likelihood_no}")

# Log-likelihood with disasters
y_pred = results_with_dis['sim_remittances'].to_numpy()
residuals = y - y_pred
num_params_2 = 6
sigma2 = np.var(residuals, ddof=num_params_2)
log_likelihood_with = -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(sigma2) - np.sum(residuals**2) / (2 * sigma2)
print(f"Log-Likelihood with disasters: {log_likelihood_with}")

# Compute AIC and BIC
aic_no = aic(log_likelihood_no, n, num_params_1)
aic_with = aic(log_likelihood_with, n, num_params_2)
bic_no = bic(log_likelihood_no, n, num_params_1)
bic_with = bic(log_likelihood_with, n, num_params_2)

print(f"Model without disasters - AIC: {aic_no}, BIC: {bic_no}")
print(f"Model with disasters - AIC: {aic_with}, BIC: {bic_with}")

if aic_with < aic_no and bic_with < bic_no:
    print("The model with disasters is preferred despite extra parameters.")
else:
    print("The model without disasters might be better considering complexity.")

# Log-likelihood values

df_diff = num_params_2 - num_params_1  # Difference in number of parameters

# Likelihood Ratio Test
LR_stat = -2 * (log_likelihood_no - log_likelihood_with)
p_value = chi2.sf(LR_stat, df_diff)

print(f"Likelihood Ratio Test Statistic: {LR_stat}, P-value: {p_value}")

if p_value < 0.05:
    print("Model 2 significantly improves the fit.")
else:
    print("Extra parameters in Model 2 are not justified.")