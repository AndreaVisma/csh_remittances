
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
param_albania  = -2.2
dem_params = [param_nta, param_stay, param_fam, param_gdp, param_close, param_albania]
eq_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
dr_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
fl_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
st_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
tot_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
dict_dis_par = dict(zip(['eq', 'dr', 'fl', 'st', 'tot'], [eq_par, dr_par, fl_par, st_par, tot_par]))

### compute disasters effect
df_dis = compute_disasters_theta(df, dict_dis_par)

### simulate one country without disasters
nep_res = simulate_one_country_no_disasters(country = "Bangladesh", dem_params = dem_params,
                                            df_rem_group = df_rem_group, df_ag_long = df_ag_long,
                                            plot = True, disable_progress = False)
nep_res['sim_remittances'] = nep_res.simulated_senders * rem_amount
goodness_of_fit_results(nep_res)

### simulate one country with disasters
nep_res = simulate_one_country_with_disasters(df_dis, "Bangladesh", True)
nep_res['sim_remittances'] = nep_res.simulated_senders * rem_amount
goodness_of_fit_results(nep_res)

### simulate all countries (deterministic)
df_res = simulate_all_countries_deterministic_no_dis(dem_params = dem_params, df_rem_group = df_rem_group, df_ag_long = df_ag_long)

df_res['sim_remittances'] = df_res.simulated_senders * rem_amount
plot_everything_about_results(df_res)
plot_country_mean(df_res)

df_res_small = df_res[df_res['remittances'] > 10_000]
plot_everything_about_results(df_res_small)
plot_country_mean(df_res_small)

## plot how close the real percentage of senders and the simulated one are over time
nep_res = pct_senders_country("Philippines", plot = True)
