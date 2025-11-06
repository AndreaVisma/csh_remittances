

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from pandas.tseries.offsets import MonthEnd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from italy.simulation.func.goodness_of_fit import goodness_of_fit_results
from utils import zero_values_before_first_positive_and_after_first_negative

param_stay = 0

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors_2210.pkl"
df = pd.read_pickle(diasporas_file)
df = df.dropna()
df['year'] = df.date.dt.year

###gdp to infer remittances amount
df_gdp = pd.read_pickle("c:\\data\\economic\\gdp\\annual_gdp_per_capita_splined.pkl")
df = df.merge(df_gdp, on=['destination', 'date'], how='left')
df['rem_amount'] = 0.15 * df.gdp /12

## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags_NEW.pkl")

## load remittances
#ITA
df_rem_ita = pd.read_parquet("C:\\Data\\my_datasets\\italy\\simulation_data.parquet")
df_rem_ita['destination'] = 'Italy'
df_rem_ita.rename(columns = {"country": 'origin'}, inplace = True)
df_rem_ita = df_rem_ita[~df_rem_ita[["date", "origin"]].duplicated()][
    ["date", "origin", "destination", "remittances"]]
# PHIL
df_rem_phil = pd.read_pickle("C:\\Data\\remittances\\Philippines\\phil_remittances_detail.pkl")
# PAK
df_rem_pak = pd.read_pickle("C:\\Data\\remittances\\Pakistan\\pak_remittances_detail.pkl")
# GUA
df_rem_gua = pd.read_pickle("C:\\Data\\remittances\\Guatemala\\gua_remittances_detail.pkl")
# GUA
df_rem_nic = pd.read_pickle("C:\\Data\\remittances\\Nicaragua\\nic_remittances_detail.pkl")
# MEX
df_rem_mex = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_renamed.xlsx")[["date", "total_mln"]]
df_rem_mex['date'] = pd.to_datetime(df_rem_mex['date'], format="%Y%m") + MonthEnd(0)
df_rem_mex['origin'] = "Mexico"
df_rem_mex['destination'] = "USA"
df_rem_mex.rename(columns = {'total_mln' : 'remittances'}, inplace = True)
df_rem_mex['remittances'] *= 1_000_000

df_rem = pd.concat([df_rem_ita, df_rem_phil, df_rem_mex, df_rem_pak, df_rem_gua, df_rem_nic])
df_rem['date'] = pd.to_datetime(df_rem.date)
df_rem.sort_values(['origin', 'date'], inplace=True)
df_rem_group = df_rem.copy()
df_rem_group['year'] = df_rem_group["date"].dt.year

df = df.merge(df_rem, on =['date', 'origin', 'destination'], how = 'left')
unique_pairs = df[['date', 'origin', 'destination']].drop_duplicates()
sampled_pairs = unique_pairs.sample(frac=0.8, random_state=42)
df_sampled = df.merge(sampled_pairs, on=['date', 'origin', 'destination'], how='inner')

df_saved = df.copy()
not_sampled_pairs = unique_pairs.merge(sampled_pairs, on=['date', 'origin', 'destination'], how='outer', indicator=True)
not_sampled_pairs = not_sampled_pairs[not_sampled_pairs['_merge'] == 'left_only'].drop(columns=['_merge'])
df_not_sampled = df_saved.merge(not_sampled_pairs, on=['date', 'origin', 'destination'], how='inner')

######## functions
def simulate_row_grouped_deterministic(row, separate_disasters=False):
    # Total number of agents for this row
    n_people = row['n_people']

    if row["nta"] != 0:
        if separate_disasters:
            theta = constant + (param_nta * (row['nta'])) \
                    + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                    + (row['eq_score']) + (row['fl_score']) + (row['st_score']) + (row['dr_score'])
        else:
            theta = constant + (param_nta * (row['nta'])) \
                    + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                    + (row['tot_score'])
        # Compute remittance probability using the logistic transformation.
        p = 1 / (1 + np.exp(-theta))
    else:
        p = 0

    total_senders = int(p * n_people)

    return total_senders

################### run functions
def get_df_countries(df):
    ita_origin_countries = (df[df.destination == "Italy"]['origin'].unique().tolist())
    try:
        ita_origin_countries.remove("Cote d'Ivoire")
    except:
        pass
    ita_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.destination == "Italy")].origin.unique().tolist()
    ita_all_countries = list(set(ita_origin_countries).intersection(set(ita_countries_high_remittances)))

    phil_dest_countries = (df[df.origin == "Philippines"]['destination'].unique().tolist())
    phil_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.origin == "Philippines")].destination.unique().tolist()
    phil_all_countries = list(set(phil_dest_countries).intersection(set(phil_countries_high_remittances)))

    pak_dest_countries = (df[df.origin == "Pakistan"]['destination'].unique().tolist())
    pak_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.origin == "Pakistan")].destination.unique().tolist()
    pak_all_countries = list(set(pak_dest_countries).intersection(set(pak_countries_high_remittances)))

    nic_dest_countries = (df[df.origin == "Nicaragua"]['destination'].unique().tolist())
    nic_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.origin == "Nicaragua")].destination.unique().tolist()
    nic_all_countries = list(set(nic_dest_countries).intersection(set(nic_countries_high_remittances)))

    countries_ita = ita_all_countries
    countries_phil = phil_all_countries
    countries_pak = pak_all_countries
    countries_nic = nic_all_countries

    # df country
    df_country_ita = df.query(f"""`origin` in {countries_ita} and `destination` == 'Italy'""")
    df_country_phil = df.query(f"""`origin` == 'Philippines' and `destination` in {countries_phil}""")
    df_country_pak = df.query(f"""`origin` == 'Pakistan' and `destination` in {countries_pak}""")
    df_country_mex = df.query(f"""`origin` == 'Mexico' and `destination` == 'USA'""")
    df_country_nic = df.query(f"""`origin` == 'Nicaragua' and `destination` in {countries_nic}""")
    df_country_gua = df.query(f"""`origin` == 'Guatemala' and `destination` == 'USA'""")

    df_country_ita = df_country_ita[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year', 'origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_phil = df_country_phil[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_pak = df_country_pak[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_nic = df_country_nic[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_mex = df_country_mex[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_gua = df_country_gua[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()

    df_countries = pd.concat([df_country_ita, df_country_mex, df_country_gua, df_country_nic, df_country_pak, df_country_phil])
    return df_countries

def plot_country_mean(df, two_countries = False):
    if two_countries:
        df_mean_ita = df[df.destination == 'Italy'][['origin', 'remittances', 'sim_remittances']].groupby(['origin']).mean().reset_index()
        df_mean_phil = df[df.origin == 'Philippines'][['destination', 'remittances', 'sim_remittances']].groupby(['destination']).mean().reset_index()
        df_mean_pak = df[df.origin == 'Pakistan'][['destination', 'remittances', 'sim_remittances']].groupby(
            ['destination']).mean().reset_index()
        df_mean_nic = df[df.origin == 'Nicaragua'][['destination', 'remittances', 'sim_remittances']].groupby(
            ['destination']).mean().reset_index()
        df_mean_mex = df[df.origin.isin(["Mexico", "Guatemala"])][['origin', 'remittances', 'sim_remittances']].groupby(
            ['origin']).mean().reset_index()
        df_mean = pd.concat([df_mean_ita, df_mean_phil, df_mean_mex, df_mean_pak, df_mean_nic])
        fig = go.Figure()

        # Add traces with loop
        for df, color, name, text_col, prefix in zip(
                [df_mean_ita, df_mean_phil, df_mean_mex, df_mean_pak, df_mean_nic],
                ['blue', 'red', 'orange', 'green', 'pink'],
                ['From Italy', 'To Philippines', 'from USA', 'To Pakistan', 'To Nicaragua'],
                ['origin', 'destination', 'origin', 'destination', 'destination'],
                ['Origin', 'Destination', 'origin', 'destination', 'destination']
        ):
            fig.add_trace(go.Scatter(
                x=df['remittances'],
                y=df['sim_remittances'],
                mode='markers',
                name=name,
                marker=dict(color=color, size = 10),
                text=df[text_col],
                hovertemplate=f'{prefix}: %{{text}}<br>Remittances: %{{x}}<br>Simulated: %{{y}}'
            ))

        # Add 1:1 line
        max_val = max(df_mean_ita['remittances'].max(), df_mean_phil['remittances'].max(), df_mean_mex['remittances'].max())
        fig.add_trace(go.Scatter(
            x=np.linspace(0, max_val, 100),
            y=np.linspace(0, max_val, 100),
            mode='lines',
            name='1:1 Line',
            line=dict(color='gray', dash='dash')
        ))

        fig.update_layout(
            title='Simulated vs Actual Remittances',
            xaxis=dict(title='Actual Remittances (log scale)'),
            yaxis=dict(title='Simulated Remittances (log scale)'),
            legend=dict(title='Legend'),
            template='plotly_white'
        )
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    else:
        df_mean = df[['origin', 'remittances', 'sim_remittances']].groupby(['origin']).mean().reset_index()
        fig = px.scatter(df_mean, x = 'remittances', y = 'sim_remittances',
                         color = 'origin', log_x=True, log_y=True)
        fig.add_scatter(x=np.linspace(0, df_mean.remittances.max(), 100),
                        y=np.linspace(0, df_mean.remittances.max(), 100))
    fig.show()
    goodness_of_fit_results(df_mean)

def calculate_tot_score(emdat_ita, height, shape, shift):
    global dict_scores
    dict_scores = dict(zip([x for x in range(12)],
                           zero_values_before_first_positive_and_after_first_negative(
                               [height + shape * np.sin((np.pi / 6) * (x+shift)) for x in range(1, 13)])))
    for x in range(12):
        emdat_ita[f"tot_{x}"] = emdat_ita[[f"eq_{x}",f"dr_{x}",f"fl_{x}",f"st_{x}"]].sum(axis =1) * dict_scores[x]
    emdat_ita["tot_score"] = emdat_ita[[x for x in emdat_ita.columns if "tot" in x]].sum(axis =1)
    return emdat_ita[['date', 'origin', 'tot_score']]

def check_params_combo_faster(df_countries, height, shape, shift, rem_pct, plot = True):

    emdat_ita = emdat[emdat.origin.isin(df_countries.origin.unique())].copy()
    emdat_ita = calculate_tot_score(emdat_ita, height, shape, shift)
    try:
        df_countries.drop(columns = 'tot_score', inplace = True)
    except:
        pass
    df_countries = df_countries.merge(emdat_ita, on=['origin', 'date'], how='left')
    df_countries['tot_score'].fillna(0, inplace = True)
    df_countries['rem_amount'] = rem_pct * df_countries['gdp'] / 12

    df_countries['theta'] = constant + (param_nta * (df_countries['nta'])) + (param_inc * (df_countries["gdp_origin_norm"])) \
                    + (param_asy * df_countries['asymmetry']) + (param_gdp * df_countries['relative_diff']) \
                    + (df_countries['tot_score'])
    df_countries['probability'] = 1 / (1 + np.exp(-df_countries["theta"]))
    df_countries.loc[df_countries.nta == 0, 'probability'] = 0
    df_countries['sim_senders'] = (df_countries['probability'] * df_countries['n_people']).astype(int)
    df_countries['sim_remittances'] = df_countries['sim_senders'] * df_countries['rem_amount']

    remittance_per_period = df_countries.groupby(['date', 'origin', 'destination'])[['sim_remittances', 'sim_senders']].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")

    if plot:
        goodness_of_fit_results(remittance_per_period)

        plot_country_mean(remittance_per_period, two_countries=True)

    return remittance_per_period

################
def error_function(params):
    global param_nta, param_asy, param_gdp, height, shape, shift, constant
    param_nta, param_asy, param_gdp, param_inc, height, shape, shift, constant, rem_pct = params

    res = check_params_combo_faster(df_countries, height, shape, shift, rem_pct, plot=False)
    res['error'] = (res['sim_remittances'] - res['remittances']) / 1e9
    res['error'] = np.square(res['error'])
    return res['error'].sum()

df_countries = get_df_countries(df_sampled)

params = [np.float64(1.2),
 np.float64(-5),
 np.float64(0.6),
 np.float64(-3),
 np.float64(0.15),
 np.float64(0.23),
 np.float64(-1),
 np.float64(0.1),
 np.float64(0.2)]

param_nta, param_asy, param_gdp, param_inc, height, shape, shift, constant, rem_pct = params
results = check_params_combo_faster(df_countries, height, shape, shift, rem_pct, plot = True)

res = minimize(
    lambda x: error_function(x),
    x0 = params,
    bounds= [(0.1,3),(-10,-4),(0,5),(-4, 0),(-0.5,1),(-0.5,1),(-2,2),(-2,2), (0.12, 0.3)],
    method="L-BFGS-B",
    options={'disp': True}
)

dict_best = dict(zip(['nta', 'asy', 'gdp', 'income_origin', 'height', 'shape', 'shift', 'constant', 'rem_pct'], res.x))
for k, v in dict_best.items():
    print(f"{k}:{v}")
print("Predicted error:", res.fun)

#####################
def return_train_test_result(params):
    global param_nta, param_asy, param_gdp, height, shape, shift, constant, rem_pct
    param_nta, param_asy, param_gdp, height, shape, shift, constant, rem_pct = params

    df_countries = get_df_countries(df_sampled)
    emdat_ita = emdat[emdat.origin.isin(df_countries.origin.unique())].copy()
    emdat_ita = calculate_tot_score(emdat_ita, height, shape, shift)
    try:
        df_countries.drop(columns = 'tot_score', inplace = True)
    except:
        pass
    df_countries = df_countries.merge(emdat_ita, on=['origin', 'date'], how='left')
    df_countries['tot_score'].fillna(0, inplace = True)
    df_countries['rem_amount'] = rem_pct * df_countries['gdp'] / 12

    df_countries['theta'] = constant + (param_nta * (df_countries['nta'])) \
                            + (param_asy * df_countries['asymmetry']) + (param_gdp * df_countries['relative_diff']) \
                            + (df_countries['tot_score'])
    df_countries['probability'] = 1 / (1 + np.exp(-df_countries["theta"]))
    df_countries.loc[df_countries.nta == 0, 'probability'] == 0
    df_countries['sim_senders'] = (df_countries['probability'] * df_countries['n_people']).astype(int)
    df_countries['sim_remittances'] = df_countries['sim_senders'] * df_countries['rem_amount']

    remittance_per_period_1 = df_countries.groupby(['date', 'origin', 'destination'])[['sim_remittances', 'sim_senders']].sum().reset_index()
    remittance_per_period_1 = remittance_per_period_1.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")
    remittance_per_period_1['type'] = 'train'

    df_countries = get_df_countries(df_not_sampled)
    emdat_ita = emdat[emdat.origin.isin(df_countries.origin.unique())].copy()
    emdat_ita = calculate_tot_score(emdat_ita, height, shape, shift)
    try:
        df_countries.drop(columns = 'tot_score', inplace = True)
    except:
        pass
    df_countries = df_countries.merge(emdat_ita, on=['origin', 'date'], how='left')
    df_countries['tot_score'].fillna(0, inplace = True)
    df_countries['rem_amount'] = rem_pct * df_countries['gdp'] / 12

    df_countries['sim_senders'] = df_countries.apply(simulate_row_grouped_deterministic, axis=1)
    df_countries['sim_remittances'] = df_countries['sim_senders'] * df_countries['rem_amount']

    remittance_per_period = df_countries.groupby(['date', 'origin', 'destination'])[['sim_remittances', 'sim_senders']].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")
    remittance_per_period['type'] = 'test'

    return pd.concat([remittance_per_period_1, remittance_per_period])

def plot_train_test(df_test):

    train = df_test[df_test.type == 'train']
    test = df_test[df_test.type == 'test']

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(train['remittances'], train['sim_remittances'], alpha=0.5, label = 'Training sample')
    ax.scatter(test['remittances'], test['sim_remittances'], alpha=0.5, label='Test sample', marker = 'x')
    lims = [0, train['remittances'].max()]
    ax.plot(lims, lims, 'k-', alpha=1, zorder=1)
    plt.xlabel('Observed Remittances')
    plt.ylabel('Simulated Remittances')
    plt.title("Calibration results")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    # Training sample plot
    axs[0].scatter(train['remittances'], train['sim_remittances'], alpha=0.5, label='Training sample')
    lims = [0, max(train['remittances'].max(), train['sim_remittances'].max())]
    axs[0].plot(lims, lims, 'k-', alpha=1, zorder=1)
    axs[0].set_title("Training Sample")
    axs[0].set_xlabel("Observed Remittances")
    axs[0].set_ylabel("Simulated Remittances")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].grid(True)

    # Test sample plot
    axs[1].scatter(test['remittances'], test['sim_remittances'], alpha=0.5, label='Test sample', marker='x', color = 'orange')
    lims = [0, max(test['remittances'].max(), test['sim_remittances'].max())]
    axs[1].plot(lims, lims, 'k-', alpha=1, zorder=1)
    axs[1].set_title("Test Sample")
    axs[1].set_xlabel("Observed Remittances")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle("Calibration Results")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('.\plots\\for_paper\\calibration.svg', bbox_inches='tight')
    plt.show(block = True)

params = [x for x in res.x]
# params = [1.89, -9.466, 7, 0.27, 0.39, -2, -0.84, 0.11]
param_nta, param_asy, param_gdp, param_inc, height, shape, shift, constant, rem_pct = params
res = check_params_combo_faster(get_df_countries(df_not_sampled), height, shape, shift, rem_pct, plot=True)

df_test = return_train_test_result(params)
plot_train_test(df_test)
test_plot = df_test[df_test.type == 'test']

fig, ax = plt.subplots(figsize=(12, 9))
ax.scatter(test_plot['remittances'], test_plot['sim_remittances'], alpha=0.5, label='Test sample', marker = 'o')
lims = [0, test_plot['remittances'].max()]
ax.plot(lims, lims, 'k-', alpha=1, zorder=1)
# plt.xlabel('Observed Remittances')
# plt.ylabel('Simulated Remittances')
# plt.title("Calibration results")
plt.yscale('log')
plt.xscale('log')
# plt.legend()
plt.grid(True)
fig.savefig('.\plots\\for_paper\\CALIBRATION.svg', bbox_inches = 'tight')
plt.show(block = True)


fig = px.scatter(df_test, x="remittances", y="sim_remittances", trendline="ols", color = 'type')
fig.show()

results = px.get_trendline_results(fig)
results.px_fit_results.iloc[0].summary()


##########################################
# param_names = {0:'nta', 1:'asy', 2:'gdp', 3:'height', 4:'shape', 5:'shift', 6:'constant'}
#
# for i in tqdm(range(1000)):
#     param_to_change = random.randint(0, 6)
#     change = random.choice([-0.05, 0.05])
#     params[param_to_change] = params[param_to_change] + change
#     error_new = error_function(params)
#
#     if error_new < error_old:
#         print(f"Iter {i}, found better model, changed param nr {param_names[param_to_change]}")
#         t = i
#         error_old = error_new
#     else:
#         params[param_to_change] = params[param_to_change] - change
#
#     if i - t > 50:
#         print("20 iters and no change, stopping ...")
#         break
#
# best_params = params

##############
# build parameter space
# results_list = []
# for i in tqdm(range(400)):
#     param_nta = uniform(0.5,3)
#     param_asy = uniform(-10,-5)
#     param_gdp = uniform(7,13)
#     height = uniform(-0.5,1)
#     shape = uniform(-0.5,1)
#     constant = uniform(-2,2)
#     shift = uniform(-2,2)
#
#     if abs(height) > abs(shape):
#         print("skipped one sim")
#         continue
#
#     error = error_function([param_nta, param_asy, param_gdp, height, shape, shift, constant])
#
#     dict_params = {"nta": [param_nta],
#                    "asy": [param_asy],
#                    "stay": [param_stay],
#                    "gdp": [param_gdp],
#                    "height": [height],
#                    "shape": [shape],
#                    "shift" : [shift],
#                    "constant": [constant],
#                    "squared_err" : [error]}
#     df_res = pd.DataFrame.from_dict(dict_params)
#     results_list.append(df_res)
#
# df_results = pd.concat(results_list)
# df_results.sort_values('squared_err', inplace = True)
#
# df_results.to_excel("C://Users//Andrea Vismara//Downloads//sim_errors_0707.xlsx", index = False)
#############################