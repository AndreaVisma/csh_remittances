

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

#### simulate
df_dis = compute_disasters_theta(df, dict_dis_par)
dem_params = [param_nta, param_stay, param_fam, param_gdp, param_close, param_albania]

results_with_dis = simulate_all_countries(df_dis, df, dem_params, df_rem_group, df_ag_long, disasters = True)
results_with_dis['sim_remittances'] = results_with_dis['simulated_senders'] * rem_amount

####
prueba = results_with_dis[['remittances', 'simulated_senders', 'country', 'sim_remittances']].groupby('country').mean().reset_index()
pop_mean = df_rem_group[['country', 'population']].groupby('country').mean().reset_index()
prueba = prueba.merge(pop_mean, on = 'country', how = 'left')

prueba['pct_senders'] = 100 * prueba['simulated_senders'] / prueba['population']
prueba['avg_remittance_value'] = prueba['remittances'] / prueba['simulated_senders']
prueba.sort_values('remittances', ascending = False, inplace = True)
prueba.to_excel("C:\\Data\\remittances\\italy\\prueba_fuego_model.xlsx", index = False)

### avg remittance value
fig, ax = plt.subplots()
prueba['avg_remittance_value'].hist(bins = 100, ax=ax)
plt.show(block = True)

