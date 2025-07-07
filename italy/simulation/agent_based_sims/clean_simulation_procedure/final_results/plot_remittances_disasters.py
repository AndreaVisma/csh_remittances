
import pandas as pd
import numpy as np
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

nep_res_no = simulate_one_country_no_disasters(country = "Bangladesh", dem_params = dem_params,
                                            df_rem_group = df_rem_group, df_ag_long = df_ag_long,
                                            plot = True, disable_progress = False)
nep_res_no['sim_remittances_no'] = nep_res_no.simulated_senders * rem_amount * 1.6

nep_res = simulate_one_country_with_disasters(df_dis, df, country = "Bangladesh", plot = True)
nep_res['sim_remittances'] = nep_res.simulated_senders * rem_amount * 1.6
nep_res = nep_res.merge(nep_res_no[['date', 'sim_remittances_no']], on = 'date')
df_plot = nep_res.copy()
df_plot['date'] = pd.to_datetime(df_plot['date'])
df_plot = df_plot[df_plot['date'].dt.year >= 2015]

df_nat = pd.read_csv("C:\\Data\\my_datasets\\weekly_remittances\\weekly_disasters.csv")
df_nat["week_start"] = pd.to_datetime(df_nat["week_start"])
df_nat["year"] = df_nat.week_start.dt.year

df_pop_country = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")
df_nat = df_nat.merge(df_pop_country, on = ['country', 'year'], how = 'left')
df_nat['total_affected'] = 100 * df_nat['total_affected'] / df_nat["population"]
df_nat = df_nat[["week_start", "total_affected", "total_damage", "country", "type"]]

# Preprocess data to align disasters with remittance months
# ---------------------------------------------------------
# Convert week_start to monthly frequency and aggregate by disaster type
df_nat_monthly = (
    df_nat.groupby(['country', 'type', pd.Grouper(key='week_start', freq='M')])
    .agg({'total_affected': 'sum', 'total_damage': 'sum'})
    .reset_index()
    .rename(columns={'week_start': 'date'})
)
df_nat_monthly = df_nat_monthly[(df_nat_monthly.country == "Bangladesh") & (df_nat_monthly.type.isin(['Flood', 'Storm']))]
df_nat_monthly["date"] = pd.to_datetime(df_nat_monthly["date"])
df_nat_monthly = df_nat_monthly[df_nat_monthly.date.dt.year >= 2015]
df_nat_monthly = df_nat_monthly[df_nat_monthly.date.dt.year < 2024]
df_nat_monthly.sort_values('date', inplace = True)

disaster_periods = [
    ('2015-05-01', '2015-05-31', 'Storm'),
    ('2016-07-01', '2016-07-31', 'Flood'),
    ('2017-05-01', '2017-05-31', 'Storm'),
    ('2017-08-01', '2017-08-30', 'Flood'),
    ('2019-05-01', '2019-07-31', 'Flood'),
    ('2020-05-01', '2020-05-31', 'Storm'),
    ('2020-06-01', '2020-09-30', 'Flood'),
    ('2021-05-01', '2021-05-31', 'Storm'),
    ('2022-05-01', '2022-08-31', 'Flood'),
    ('2021-07-01', '2021-07-31', 'Storm'),
    ('2023-10-01', '2023-10-31', 'Storm')

]

# Define colors for each disaster type
disaster_colors = {
    'Flood': 'orange',
    'Storm': 'brown'
}

# Convert dates to datetime if necessary
df_plot['date'] = pd.to_datetime(df_plot['date'])

# Plot the lines
plt.figure(figsize=(6, 6))
plt.plot(df_plot['date'], df_plot['remittances'], label='Observed', linestyle='-', color='blue', linewidth=2.5)
plt.plot(df_plot['date'], df_plot['sim_remittances'], label='Simulated, considering disasters', linestyle='-', color='red', linewidth=2.5)
plt.plot(df_plot['date'], df_plot['sim_remittances_no'], label='Simulated, not considering disasters', linestyle='-', color='green', linewidth=2.5)

# Highlight disaster periods with different colors
legend_labels = {}  # To track unique legend labels

for start, end, disaster_type in disaster_periods:
    color = disaster_colors.get(disaster_type, 'gray')  # Default to gray if type is unknown
    label = disaster_type if disaster_type not in legend_labels else ""  # Avoid duplicate legend labels
    plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=color, alpha=0.4, label=label, ls = "--")
    legend_labels[disaster_type] = True

# Formatting
plt.xlabel('Date')
plt.ylabel('Remittances (USD)')
plt.title('Observed vs Simulated Remittances with Disaster Events')
plt.legend().remove()
plt.grid(True)
plt.savefig("C:\\Users\\Andrea Vismara\\Desktop\\papers\\remittances\\charts\\remittances_disasters_bangladesh.svg")
plt.show(block = True)
