
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time
import itertools
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from random import sample
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)

## pair of countries
origin, destination = "Philippines", "Japan"

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_interpolated.pkl"
df = pd.read_pickle(diasporas_file)
df = df[df.n_people > 0]

##exponential betas for years of stay
df_betas = pd.read_pickle("C:\\Data\\migration\\simulations\\exponential_betas.pkl")

## family asymmetry
asymmetry_file = "C:\\Data\\migration\\bilateral_stocks\\pyramid_asymmetry_beginning_of_the_year.pkl"
asy_df = pd.read_pickle(asymmetry_file)

## diaspora growth rates
growth_rates = pd.read_pickle("C://data//migration//stock_pct_change.pkl")

## gdp differential
df_gdp = (pd.read_pickle("c:\\data\\economic\\gdp\\annual_gdp_deltas.pkl"))

##nta accounts
df_nta = pd.read_pickle("C:\\Data\\economic\\nta\\processed_nta.pkl")
nta_dict = {}

## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags_NEW.pkl")


#########################################
#########################################
# Sample parameters
param_nta = 1
param_stay = -0.2
param_asy = -3.5
param_gdp = 0.5
fixed_remittance = 1100  # Amount each sender sends x

## load italy & Philippines remittances
df_rem_ita = pd.read_parquet("C:\\Data\\my_datasets\\italy\\simulation_data.parquet")
df_rem_ita['destination'] = 'Italy'
df_rem_ita.rename(columns = {"country": 'origin'}, inplace = True)
df_rem_ita = df_rem_ita[~df_rem_ita[["date", "origin"]].duplicated()][
    ["date", "origin", "destination", "remittances"]]
df_rem_phil = pd.read_pickle("C:\\Data\\remittances\\Philippines\\phil_remittances_detail.pkl")
df_rem = pd.concat([df_rem_ita, df_rem_phil])
df_rem['date'] = pd.to_datetime(df_rem.date)
df_rem.sort_values(['origin', 'date'], inplace=True)
df_rem_group = df_rem.copy()
df_rem_group['year'] = df_rem_group["date"].dt.year

##### disasters parameters

dis_params = pd.read_excel("C:\\Data\\my_datasets\\disasters\\disasters_params.xlsx", sheet_name="Sheet2").dropna()

def sin_function(a,b,c,x):
    return a * np.sin((np.pi/6) * x) + b * np.sin((np.pi/3) * x) + c

def sin_function_simple(a,c,x):
    return a + c * np.sin((np.pi/6) * x)


def zero_values_before_first_positive_and_after_first_negative(lst):
    modified = lst.copy()
    # Find first positive
    first_positive = next((i for i, x in enumerate(lst) if x > 0), None)

    if first_positive is not None:
        # Zero before first positive
        for i in range(first_positive):
            modified[i] = 0

        # Find first negative AFTER the first positive
        first_negative_after = next(
            (i for i, x in enumerate(lst[first_positive:], start=first_positive) if x < 0),
            None
        )

        if first_negative_after is not None:
            # Zero positives after first negative encountered post-positive
            for i in range(first_negative_after, len(modified)):
                if modified[i] > 0:
                    modified[i] = 0

    return modified
def disaster_score_function(disasters = ['tot'], simple = True):
    global dict_dis_par
    dict_dis_par = {}
    for dis in disasters:
        if not simple:
            a,b,c = dis_params[dis]
            values = [sin_function(a,b,c,x) for x in np.linspace(0, 11, 12)]
            values = zero_values_before_first_positive_and_after_first_negative(values.copy())
        else:
            a,c = dis_params[dis]
            values = [sin_function_simple(a,c,x) for x in np.linspace(0, 11, 12)]
            values = zero_values_before_first_positive_and_after_first_negative(values.copy())
        dict_dis_par[dis] = values
    return dict_dis_par

dict_dis_par = disaster_score_function(disasters = ['tot'], simple=True)

####################
def compute_disasters_scores(df, dict_dis_par):
    df_dis = df.copy()
    df_dis = df_dis.drop_duplicates(subset=["date", "origin"])
    for col in ['eq', 'dr', 'fl', 'st']:
        for shift in [int(x) for x in np.linspace(1, 11, 11)]:
            g = df_dis.groupby('origin', group_keys=False)
            g = g.apply(lambda x: x.set_index(['date', 'origin'])[col]
                        .shift(shift).reset_index(drop=True)).fillna(0)
            df_dis[f'{col}_{shift}'] = g.iloc[0]
            df_dis['tot'] = df_dis['fl'] + df_dis['eq'] + df_dis['st'] + df_dis['dr']
    for shift in [int(x) for x in np.linspace(1, 11, 11)]:
        df_dis[f'tot_{shift}'] = df_dis[f'fl_{shift}'] + df_dis[f'eq_{shift}'] + df_dis[f'st_{shift}'] + df_dis[
            f'dr_{shift}']
    df_dis.rename(columns={'eq': 'eq_0', 'st': 'st_0', 'fl': 'fl_0', 'dr': 'dr_0', 'tot': 'tot_0'}, inplace=True)
    required_columns = ['date', 'origin'] + \
                       [f"{disaster}_{i}" for disaster in ['eq', 'dr', 'fl', 'st', 'tot']
                        for i in range(12)]
    missing_cols = [col for col in required_columns if col not in df_dis.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    for disaster in ['eq', 'dr', 'fl', 'st']:
        params = dict_dis_par.get(disaster)
        if not params or len(params) != 12:
            raise ValueError(f"Need exactly 12 parameters for {disaster}")
        disaster_cols = [f"{disaster}_{i}" for i in range(12)]
        weights = np.array([params[i] for i in range(12)])
        impacts = df_dis[disaster_cols].values.dot(weights)
        df_dis[f"{disaster}_score"] = impacts
    return df_dis
def parse_age_group(age_group_str):
      """Helper function to parse age_group.
         This expects strings like "20-24". """
      lower, upper = map(int, age_group_str.split('-'))
      return lower, upper

def simulate_row_grouped_deterministic(row, separate_disasters = False, group_size=25):
    # Total number of agents for this row
    n_people = int(row['n_people']) // group_size

    # Get lower and upper bounds for the age group.
    lower_age, upper_age = parse_age_group(row['age_group'])

    # Simulate individual ages uniformly within the 5-year range
    # +1 in randint since upper bound is exclusive.
    ages = np.random.randint(lower_age, upper_age + 1, size=n_people)

    # Map the simulated ages to nta values using the dictionary.
    # We assume every age in the simulated sample has an entry in nta_dict.
    nta_values = np.array([nta_dict[age] for age in ages])

    # Simulate years of stay for each agent using the beta parameter.
    yrs_stay = np.random.exponential(scale=row['beta_estimate'], size=n_people).astype(int)

    # Calculate theta for each individual:
    # Here, asymmetry and gdp_diff (and even the beta from the growth rate) are constant for all individuals in the row.
    if separate_disasters:
        theta = (param_nta * nta_values) + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['eq_score']) + (row['fl_score']) + (row['st_score']) + (row['dr_score'])
    else:
        theta = (param_nta * nta_values) + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['tot_score'])

    # Compute remittance probability using the logistic transformation.
    p = 1 / (1 + np.exp(-theta))
    # p[nta_values == 0] = 0 # Set probability to zero if nta is zero.

    # Simulate the remittance decision (1: sends remittance, 0: does not).
    total_senders = int(sum(p)) * group_size

    # Calculate the total remitted amount for this row.
    # total_remittance = total_senders * fixed_remittance * group_size
    return total_senders
def give_everyone_fixed_probability(row, separate_disasters = False, group_size=25):

    total_senders = int(row['n_people']) * 0.6

    # Calculate the total remitted amount for this row.
    # total_remittance = total_senders * fixed_remittance * group_size
    return total_senders

#########################3
values_a = np.linspace(-1, 1, 6)
values_c = np.linspace(0, 1.5, 6)


def compute_disasters_scores_all_countries(df, values_a, values_c):
    # Compute disaster totals only once
    for i in range(12):
        df[f"tot_{i}"] = df[f"eq_{i}"] + df[f"dr_{i}"] + df[f"fl_{i}"] + df[f"st_{i}"]

    disaster_cols = [f"tot_{j}" for j in range(12)]
    df_totals = df[disaster_cols].values

    df_list = []
    for a in tqdm(values_a):
        for c in values_c:
            params = [sin_function_simple(a, c, x) for x in np.linspace(0, 11, 12)]
            params = zero_values_before_first_positive_and_after_first_negative(params)
            # Compute dot product row-wise (vectorized)
            scores = df_totals @ np.array(params)

            df_disaster = pd.DataFrame({
                'origin': df['origin'],
                'date': df['date'],
                'value_a': round(a, 2),
                'value_c': round(c, 2),
                'disaster': 'tot',
                'tot_score': scores
            })
            df_list.append(df_disaster)

    return pd.concat(df_list, ignore_index=True)

out = compute_disasters_scores_all_countries(emdat, values_a, values_c)
out.to_pickle("C:\\Data\\my_datasets\\disaster_scores_only_tot.pkl")

df_scores = pd.read_pickle("C:\\Data\\my_datasets\\disaster_scores_only_tot.pkl")

##################################
# parameter space
param_nta_space = np.linspace(0.5, 1.5, 6)
param_stay_space = np.linspace(-0.8, 0, 6)
param_asy_space = np.linspace(-4, -2, 6)
param_gdp_space = np.linspace(0, 1, 6)
fixed_remittance_space = [900,1100, 1300]  # Amount each sender sends

###########################
# check initial guess
ita_origin_countries = (df[df.destination == "Italy"]['origin'].unique().tolist())
ita_origin_countries.remove("Cote d'Ivoire")
ita_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.destination == "Italy")].origin.unique().tolist()
ita_all_countries = list(set(ita_origin_countries).intersection(set(ita_countries_high_remittances)))

phil_dest_countries = (df[df.origin == "Philippines"]['destination'].unique().tolist())
phil_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.origin == "Philippines")].destination.unique().tolist()
phil_all_countries = list(set(phil_dest_countries).intersection(set(phil_countries_high_remittances)))

param_nta = 1.2
param_stay = 0
param_asy = -2.5
param_gdp = 5
fixed_remittance_ita = 350  # Amount each sender sends
fixed_remittance_phil = 700
a = 0.7
c = 0

def plot_country_mean(df, two_countries = False):
    if two_countries:
        df_mean_ita = df[df.destination == 'Italy'][['origin', 'remittances', 'sim_remittances']].groupby(['origin']).mean().reset_index()
        df_mean_phil = df[df.origin == 'Philippines'][['destination', 'remittances', 'sim_remittances']].groupby(['destination']).mean().reset_index()
        df_mean = pd.concat([df_mean_ita, df_mean_phil])
        fig = go.Figure()

        # Add traces with loop
        for df, color, name, text_col, prefix in zip(
                [df_mean_ita, df_mean_phil],
                ['blue', 'red'],
                ['From Italy', 'To Philippines'],
                ['origin', 'destination'],
                ['Origin', 'Destination']
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
        max_val = max(df_mean_ita['remittances'].max(), df_mean_phil['remittances'].max())
        fig.add_trace(go.Scatter(
            x=np.linspace(0, max_val, 100),
            y=np.linspace(0, max_val, 100),
            mode='lines',
            name='1:1 Line',
            line=dict(color='gray', dash='dash')
        ))

        fig.update_layout(
            title='Simulated vs Actual Remittances',
            xaxis=dict(title='Actual Remittances (log scale)', type='log'),
            yaxis=dict(title='Simulated Remittances (log scale)', type='log'),
            legend=dict(title='Legend'),
            template='plotly_white'
        )
    else:
        df_mean = df[['origin', 'remittances', 'sim_remittances']].groupby(['origin']).mean().reset_index()
        fig = px.scatter(df_mean, x = 'remittances', y = 'sim_remittances',
                         color = 'origin', log_x=True, log_y=True)
        fig.add_scatter(x=np.linspace(0, df_mean.remittances.max(), 100),
                        y=np.linspace(0, df_mean.remittances.max(), 100))
    fig.show()
    goodness_of_fit_results(df_mean)

def check_initial_guess(fixed_probability = False):
    countries_ita = ita_all_countries
    countries_phil = phil_all_countries
    global nta_dict
    # df country
    df_country_ita = df.query(f"""`origin` in {countries_ita} and `destination` == 'Italy'""")
    df_country_phil = df.query(f"""`origin` == 'Philippines' and `destination` in {countries_phil}""")
    df_country_ita = df_country_ita[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    df_country_phil = df_country_phil[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    # asy
    asy_df_ita = asy_df.query(f"""`destination` == 'Italy'""")
    asy_df_phil = asy_df.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.sort_values(['origin', 'date']).sort_values(['origin', 'date']).merge(asy_df_ita[["date", "asymmetry", "origin"]],
                                  on=["date", "origin"], how='left').ffill()
    df_country_phil = df_country_phil.sort_values(['origin', 'date']).sort_values(['origin', 'date']).merge(asy_df_phil[["date", "asymmetry", "destination"]],
                                          on=["date", "destination"], how='left').ffill()
    # growth rates
    growth_rates_ita = growth_rates.query(f"""`destination` == 'Italy'""")
    growth_rates_phil = growth_rates.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.merge(growth_rates_ita[["date", "yrly_growth_rate", "origin"]],
                                  on=["date", "origin"], how='left')
    df_country_phil = df_country_phil.merge(growth_rates_phil[["date", "yrly_growth_rate", "destination"]],
                                  on=["date", "destination"], how='left')

    df_country_ita['yrly_growth_rate'] = df_country_ita['yrly_growth_rate'].bfill()
    df_country_ita['yrly_growth_rate'] = df_country_ita['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country_ita = df_country_ita.merge(df_betas, on="yrly_growth_rate", how='left')
    df_country_phil['yrly_growth_rate'] = df_country_phil['yrly_growth_rate'].bfill()
    df_country_phil['yrly_growth_rate'] = df_country_phil['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country_phil = df_country_phil.merge(df_betas, on="yrly_growth_rate", how='left')

    ##gdp diff
    df_gdp_ita = df_gdp.query(f"""`destination` == 'Italy'""")
    df_gdp_phil = df_gdp.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.merge(df_gdp_ita[["date", "gdp_diff_norm", "origin"]], on=["date", "origin"],
                                  how='left')
    df_country_ita['gdp_diff_norm'] = df_country_ita['gdp_diff_norm'].bfill()
    df_country_phil = df_country_phil.merge(df_gdp_phil[["date", "gdp_diff_norm", "destination"]], on=["date", "destination"],
                                  how='left')
    df_country_phil['gdp_diff_norm'] = df_country_phil['gdp_diff_norm'].bfill()

    ## nta
    df_nta_ita = df_nta.query(f"""`country` == 'Italy'""")[['age', 'nta']].fillna(0)
    df_nta_phil = df_nta[df_nta.country.isin(countries_phil)][['country', 'age', 'nta']].fillna(0)

    emdat_ = df_scores[df_scores.origin.isin(countries_ita + ["Philippines"])]

    for ind, row in df_nta_ita.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)
    dis_params['tot'] = [a, c]
    try:
        df_country_ita.drop(columns=f"tot_score", inplace=True)
    except:
        pass
    df_country_ita = df_country_ita.merge(
        emdat_[(emdat_.value_a == a) & (emdat_.value_c == c)]
        [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
    df_country_ita['tot_score'] = df_country_ita['tot_score'].fillna(0)
    if not fixed_probability:
        df_country_ita['sim_senders'] = df_country_ita.apply(simulate_row_grouped_deterministic, axis=1)
    else:
        df_country_ita['sim_senders'] = df_country_ita.apply(give_everyone_fixed_probability, axis=1)
    df_country_ita['sim_remittances'] = df_country_ita['sim_senders'] * fixed_remittance_ita

    list_sims = []
    for country in tqdm(list(set(df_nta_phil.country).intersection(set(phil_all_countries)))):
        df_sim = df_country_phil[df_country_phil.destination == country].copy()
        for ind, row in df_nta_phil[df_nta_phil.country == country].iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        dis_params['tot'] = [a, c]
        try:
            df_sim.drop(columns=f"tot_score", inplace=True)
        except:
            pass
        df_sim = df_sim.merge(
            emdat_[(emdat_.value_a == a) & (emdat_.value_c == c)]
            [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
        df_sim['tot_score'] = df_sim['tot_score'].fillna(0)
        if not fixed_probability:
            df_sim['sim_senders'] = df_sim.apply(simulate_row_grouped_deterministic, axis=1)
        else:
            df_sim['sim_senders'] = df_sim.apply(give_everyone_fixed_probability, axis=1)
        list_sims.append(df_sim)
    df_country_phil = pd.concat(list_sims)
    df_country_phil['sim_remittances'] = df_country_phil['sim_senders'] * fixed_remittance_phil

    df_country = pd.concat([df_country_ita, df_country_phil])
    remittance_per_period = df_country.groupby(['date', 'origin', 'destination'])['sim_remittances'].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")
    remittance_per_period['error_squared'] = (remittance_per_period['remittances'] -
                                              remittance_per_period['sim_remittances']) ** 2
    goodness_of_fit_results(remittance_per_period)

    plot_country_mean(remittance_per_period, two_countries=True)


check_initial_guess(fixed_probability=False)

##########################
values_a = [0]# np.linspace(-1, 1, 2)
values_c = [0]# np.linspace(0, 1.5, 2)
param_nta_space = [round(x,2) for x in np.linspace(0.5, 1.5, 6)]
param_stay_space = [round(x,2) for x in np.linspace(-0.8, 0, 6)]
param_asy_space = [round(x,2) for x in np.linspace(-4, -2, 6)]
param_gdp_space = [round(x,2) for x in np.linspace(0, 1, 6)]
# fixed_remittance_space = [800]  # Amount each sender sends

#############
results_list = []
n_repetitions = 6
for f in tqdm(range(n_repetitions)):
    countries_ita = sample(ita_all_countries, int(len(ita_all_countries) * 0.6))
    countries_phil = sample(phil_all_countries, int(len(phil_all_countries) * 0.6))

    global nta_dict
    # df country
    df_country_ita = df.query(f"""`origin` in {countries_ita} and `destination` == 'Italy'""")
    df_country_phil = df.query(f"""`origin` == 'Philippines' and `destination` in {countries_phil}""")
    df_country_ita = df_country_ita[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    df_country_phil = df_country_phil[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    # asy
    asy_df_ita = asy_df.query(f"""`destination` == 'Italy'""")
    asy_df_phil = asy_df.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.sort_values(['origin', 'date']).sort_values(['origin', 'date']).merge(asy_df_ita[["date", "asymmetry", "origin"]],
                                          on=["date", "origin"], how='left').ffill()
    df_country_phil = df_country_phil.sort_values(['origin', 'date']).sort_values(['origin', 'date']).merge(asy_df_phil[["date", "asymmetry", "destination"]],
                                            on=["date", "destination"], how='left').ffill()
    # growth rates
    growth_rates_ita = growth_rates.query(f"""`destination` == 'Italy'""")
    growth_rates_phil = growth_rates.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.merge(growth_rates_ita[["date", "yrly_growth_rate", "origin"]],
                                          on=["date", "origin"], how='left')
    df_country_phil = df_country_phil.merge(growth_rates_phil[["date", "yrly_growth_rate", "destination"]],
                                            on=["date", "destination"], how='left')

    df_country_ita['yrly_growth_rate'] = df_country_ita['yrly_growth_rate'].bfill()
    df_country_ita['yrly_growth_rate'] = df_country_ita['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country_ita = df_country_ita.merge(df_betas, on="yrly_growth_rate", how='left')
    df_country_phil['yrly_growth_rate'] = df_country_phil['yrly_growth_rate'].bfill()
    df_country_phil['yrly_growth_rate'] = df_country_phil['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country_phil = df_country_phil.merge(df_betas, on="yrly_growth_rate", how='left')

    ##gdp diff
    df_gdp_ita = df_gdp.query(f"""`destination` == 'Italy'""")
    df_gdp_phil = df_gdp.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.merge(df_gdp_ita[["date", "gdp_diff_norm", "origin"]], on=["date", "origin"],
                                          how='left')
    df_country_ita['gdp_diff_norm'] = df_country_ita['gdp_diff_norm'].bfill()
    df_country_phil = df_country_phil.merge(df_gdp_phil[["date", "gdp_diff_norm", "destination"]],
                                            on=["date", "destination"],
                                            how='left')
    df_country_phil['gdp_diff_norm'] = df_country_phil['gdp_diff_norm'].bfill()

    ## nta
    df_nta_ita = df_nta.query(f"""`country` == 'Italy'""")[['age', 'nta']].fillna(0)
    df_nta_phil = df_nta[df_nta.country.isin(countries_phil)][['country', 'age', 'nta']].fillna(0)

    emdat_ = df_scores[df_scores.origin.isin(countries_ita + ["Philippines"])]
    df_country_ita_ = df_country_ita.copy()
    df_country_phil_ = df_country_phil.copy()

    for param_nta in tqdm(param_nta_space):
        for param_asy in param_asy_space:
            for param_stay in param_stay_space:
                for param_gdp in param_gdp_space:
                    for fixed_remittance in fixed_remittance_space:
                        for a in values_a:
                            for c in values_c:
                                dis_params['tot'] = [a,c]
                                try:
                                    df_country_ita.drop(columns=f"tot_score", inplace=True)
                                except:
                                    pass
                                df_country_ita = df_country_ita.merge(
                                    emdat_[(emdat_.disaster == 'tot') & (emdat_.value_a == a) & (emdat_.value_c == c)]
                                    [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
                                df_country_ita['tot_score'] = df_country_ita['tot_score'].fillna(0)

                                df_country_ita['sim_senders'] = df_country_ita.apply(simulate_row_grouped_deterministic, axis=1)
                                df_country_ita['sim_remittances'] = df_country_ita.sim_senders * fixed_remittance_ita

                                list_sims = []
                                for country in tqdm(
                                        list(set(df_nta_phil.country).intersection(set(phil_all_countries)))):
                                    df_sim = df_country_phil[df_country_phil.destination == country].copy()
                                    for ind, row in df_nta_phil[df_nta_phil.country == country].iterrows():
                                        nta_dict[int(row.age)] = round(row.nta, 2)
                                    dis_params['tot'] = [a, c]
                                    try:
                                        df_sim.drop(columns=f"tot_score", inplace=True)
                                    except:
                                        pass
                                    df_sim = df_sim.merge(
                                        emdat_[(emdat_.value_a == a) & (emdat_.value_c == c)]
                                        [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
                                    df_sim['tot_score'] = df_sim['tot_score'].fillna(0)
                                    df_sim['sim_senders'] = df_sim.apply(simulate_row_grouped_deterministic, axis=1)
                                    list_sims.append(df_sim)
                                df_country_phil = pd.concat(list_sims)
                                df_country_phil['sim_remittances'] = df_country_phil['sim_senders'] * fixed_remittance_phil
                                df_country = pd.concat([df_country_ita, df_country_phil])
                                remittance_per_period = df_country.groupby(['date', 'origin', 'destination'])['sim_remittances'].sum().reset_index()
                                remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")
                                remittance_per_period['error_squared'] = (remittance_per_period['remittances'] -
                                                                          remittance_per_period['sim_remittances']) ** 2

                                dict_params = {"nta" : param_nta,
                                               "asy" : param_asy,
                                               "stay" : param_stay,
                                               "gdp" : param_gdp,
                                               "a" : a,
                                               "c" : c,
                                               "rem_value" : fixed_remittance}
                                results_run = [dict_params, remittance_per_period['error_squared'].mean()] + [f]
                                results_list.append(results_run)

                                df_country_ita = df_country_ita_.copy()
                                df_country_phil = df_country_phil_.copy()

import pickle
with open('model_results_ita_phil.pkl', 'wb') as fi:
    pickle.dump(results_list, fi)

with open('model_results_ita_phil.pkl', 'rb') as fi:
    loaded_data = pickle.load(fi)
################

min_tuple_each_run = []
for f in tqdm(range(n_repetitions)):
    sub_data = [x for x in results_list if x[2] == f]
    flattened_data = [item[1] for item in sub_data]
    min_tuple_index = flattened_data.index(min(flattened_data))
    min_tuple = sub_data[min_tuple_index]
    min_tuple_each_run.append(min_tuple)

def r_squared_all_and_mean():
    countries_ita = ita_all_countries
    countries_phil = phil_all_countries

    global nta_dict
    # df country
    df_country_ita = df.query(f"""`origin` in {countries_ita} and `destination` == 'Italy'""")
    df_country_phil = df.query(f"""`origin` == 'Philippines' and `destination` in {countries_phil}""")
    df_country_ita = df_country_ita[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    df_country_phil = df_country_phil[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    # asy
    asy_df_ita = asy_df.query(f"""`destination` == 'Italy'""")
    asy_df_phil = asy_df.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.sort_values(['origin', 'date']).merge(asy_df_ita[["date", "asymmetry", "origin"]],
                                          on=["date", "origin"], how='left').ffill()
    df_country_phil = df_country_phil.sort_values(['origin', 'date']).merge(asy_df_phil[["date", "asymmetry", "destination"]],
                                            on=["date", "destination"], how='left').ffill()
    # growth rates
    growth_rates_ita = growth_rates.query(f"""`destination` == 'Italy'""")
    growth_rates_phil = growth_rates.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.merge(growth_rates_ita[["date", "yrly_growth_rate", "origin"]],
                                          on=["date", "origin"], how='left')
    df_country_phil = df_country_phil.merge(growth_rates_phil[["date", "yrly_growth_rate", "destination"]],
                                            on=["date", "destination"], how='left')

    df_country_ita['yrly_growth_rate'] = df_country_ita['yrly_growth_rate'].bfill()
    df_country_ita['yrly_growth_rate'] = df_country_ita['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country_ita = df_country_ita.merge(df_betas, on="yrly_growth_rate", how='left')
    df_country_phil['yrly_growth_rate'] = df_country_phil['yrly_growth_rate'].bfill()
    df_country_phil['yrly_growth_rate'] = df_country_phil['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country_phil = df_country_phil.merge(df_betas, on="yrly_growth_rate", how='left')

    ##gdp diff
    df_gdp_ita = df_gdp.query(f"""`destination` == 'Italy'""")
    df_gdp_phil = df_gdp.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.merge(df_gdp_ita[["date", "gdp_diff_norm", "origin"]], on=["date", "origin"],
                                          how='left')
    df_country_ita['gdp_diff_norm'] = df_country_ita['gdp_diff_norm'].bfill()
    df_country_phil = df_country_phil.merge(df_gdp_phil[["date", "gdp_diff_norm", "destination"]],
                                            on=["date", "destination"],
                                            how='left')
    df_country_phil['gdp_diff_norm'] = df_country_phil['gdp_diff_norm'].bfill()

    ## nta
    df_nta_ita = df_nta.query(f"""`country` == 'Italy'""")[['age', 'nta']].fillna(0)
    df_nta_phil = df_nta[df_nta.country.isin(countries_phil)][['country', 'age', 'nta']].fillna(0)

    emdat_ = df_scores[df_scores.origin.isin(countries_ita + ["Philippines"])]
    dis_params['tot'] = [a, c]
    try:
        df_country_ita.drop(columns=f"tot_score", inplace=True)
    except:
        pass
    df_country_ita = df_country_ita.merge(
        emdat_[(emdat_.disaster == 'tot') & (emdat_.value_a == a) & (emdat_.value_c == c)]
        [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
    df_country_ita['tot_score'] = df_country_ita['tot_score'].fillna(0)

    df_country_ita['sim_senders'] = df_country_ita.apply(simulate_row_grouped_deterministic, axis=1)
    df_country_ita['sim_remittances'] = df_country_ita.sim_senders * fixed_remittance_ita

    list_sims = []
    for country in tqdm(
            list(set(df_nta_phil.country).intersection(set(phil_all_countries)))):
        df_sim = df_country_phil[df_country_phil.destination == country].copy()
        for ind, row in df_nta_phil[df_nta_phil.country == country].iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        dis_params['tot'] = [a, c]
        try:
            df_sim.drop(columns=f"tot_score", inplace=True)
        except:
            pass
        df_sim = df_sim.merge(
            emdat_[(emdat_.value_a == a) & (emdat_.value_c == c)]
            [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
        df_sim['tot_score'] = df_sim['tot_score'].fillna(0)
        df_sim['sim_senders'] = df_sim.apply(simulate_row_grouped_deterministic, axis=1)
        list_sims.append(df_sim)
    df_country_phil = pd.concat(list_sims)
    df_country_phil['sim_remittances'] = df_country_phil['sim_senders'] * fixed_remittance_phil
    df_country = pd.concat([df_country_ita, df_country_phil])
    remittance_per_period = df_country.groupby(['date', 'origin', 'destination'])['sim_remittances'].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")
    remittance_per_period['error_squared'] = (remittance_per_period['remittances'] -
                                              remittance_per_period['sim_remittances']) ** 2
    remittance_per_period['error'] = remittance_per_period['remittances'] - remittance_per_period['sim_remittances']
    SS_res = np.sum(np.square(remittance_per_period['error']))
    SS_tot = np.sum(np.square(remittance_per_period['remittances'] - np.mean(remittance_per_period['remittances'])))
    R_squared = 1 - (SS_res / SS_tot)
    print(f"R-squared: {round(R_squared, 3)}")
    df_mean = remittance_per_period[['origin', 'remittances', 'sim_remittances']].groupby(['origin']).mean().reset_index()
    df_mean['error'] = df_mean['remittances'] - df_mean['sim_remittances']
    SS_res = np.sum(np.square(df_mean['error']))
    SS_tot = np.sum(np.square(df_mean['remittances'] - np.mean(df_mean['remittances'])))
    R_squared = 1 - (SS_res / SS_tot)
    print(f"R-squared means: {round(R_squared, 3)}")

for min_tuple in min_tuple_each_run:
    best_params = min_tuple[0]
    print(best_params)
    print(f"Abs error: {min_tuple[1] / 1_000_000_000_000}")
    param_nta = best_params["nta"]
    param_stay = best_params["stay"]
    param_asy = best_params["asy"]
    param_gdp = best_params["gdp"]
    a = best_params["a"]
    c = best_params["c"]
    fixed_remittance = best_params["rem_value"]
    r_squared_all_and_mean()

check_initial_guess()

##########################################
# calibrate disasters
##########################################
##########################
values_a = np.linspace(-1, 1, 6)
values_c = np.linspace(0, 1.5, 6)

param_nta = best_params["nta"]
param_stay = best_params["stay"]
param_asy = best_params["asy"]
param_gdp = best_params["gdp"]

results_list = []
n_repetitions = 1
for f in tqdm(range(n_repetitions)):
    countries_ita = sample(ita_all_countries, int(len(ita_all_countries) * 1))
    countries_phil = sample(phil_all_countries, int(len(phil_all_countries) * 1))

    global nta_dict
    # df country
    df_country_ita = df.query(f"""`origin` in {countries_ita} and `destination` == 'Italy'""")
    df_country_phil = df.query(f"""`origin` == 'Philippines' and `destination` in {countries_phil}""")
    df_country_ita = df_country_ita[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    df_country_phil = df_country_phil[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    # asy
    asy_df_ita = asy_df.query(f"""`destination` == 'Italy'""")
    asy_df_phil = asy_df.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.sort_values(['origin', 'date']).merge(asy_df_ita[["date", "asymmetry", "origin"]],
                                          on=["date", "origin"], how='left').ffill()
    df_country_phil = df_country_phil.sort_values(['origin', 'date']).merge(asy_df_phil[["date", "asymmetry", "destination"]],
                                            on=["date", "destination"], how='left').ffill()
    # growth rates
    growth_rates_ita = growth_rates.query(f"""`destination` == 'Italy'""")
    growth_rates_phil = growth_rates.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.merge(growth_rates_ita[["date", "yrly_growth_rate", "origin"]],
                                          on=["date", "origin"], how='left')
    df_country_phil = df_country_phil.merge(growth_rates_phil[["date", "yrly_growth_rate", "destination"]],
                                            on=["date", "destination"], how='left')

    df_country_ita['yrly_growth_rate'] = df_country_ita['yrly_growth_rate'].bfill()
    df_country_ita['yrly_growth_rate'] = df_country_ita['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country_ita = df_country_ita.merge(df_betas, on="yrly_growth_rate", how='left')
    df_country_phil['yrly_growth_rate'] = df_country_phil['yrly_growth_rate'].bfill()
    df_country_phil['yrly_growth_rate'] = df_country_phil['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country_phil = df_country_phil.merge(df_betas, on="yrly_growth_rate", how='left')

    ##gdp diff
    df_gdp_ita = df_gdp.query(f"""`destination` == 'Italy'""")
    df_gdp_phil = df_gdp.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.merge(df_gdp_ita[["date", "gdp_diff_norm", "origin"]], on=["date", "origin"],
                                          how='left')
    df_country_ita['gdp_diff_norm'] = df_country_ita['gdp_diff_norm'].bfill()
    df_country_phil = df_country_phil.merge(df_gdp_phil[["date", "gdp_diff_norm", "destination"]],
                                            on=["date", "destination"],
                                            how='left')
    df_country_phil['gdp_diff_norm'] = df_country_phil['gdp_diff_norm'].bfill()

    ## nta
    df_nta_ita = df_nta.query(f"""`country` == 'Italy'""")[['age', 'nta']].fillna(0)
    df_nta_phil = df_nta[df_nta.country.isin(countries_phil)][['country', 'age', 'nta']].fillna(0)

    emdat_ = df_scores[df_scores.origin.isin(countries_ita + ["Philippines"])]
    df_country_ita_ = df_country_ita.copy()
    df_country_phil_ = df_country_phil.copy()

    for a in tqdm(values_a):
        for c in values_c:
            dis_params['tot'] = [a,c]
            try:
                df_country_ita.drop(columns=f"tot_score", inplace=True)
            except:
                pass
            df_country_ita = df_country_ita.merge(
                emdat_[(emdat_.disaster == 'tot') & (emdat_.value_a == a) & (emdat_.value_c == c)]
                [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
            df_country_ita['tot_score'] = df_country_ita['tot_score'].fillna(0)

            df_country_ita['sim_senders'] = df_country_ita.apply(simulate_row_grouped_deterministic, axis=1)
            df_country_ita['sim_remittances'] = df_country_ita.sim_senders * fixed_remittance_ita

            list_sims = []
            for country in list(set(df_nta_phil.country).intersection(set(phil_all_countries))):
                df_sim = df_country_phil[df_country_phil.destination == country].copy()
                for ind, row in df_nta_phil[df_nta_phil.country == country].iterrows():
                    nta_dict[int(row.age)] = round(row.nta, 2)
                dis_params['tot'] = [a, c]
                try:
                    df_sim.drop(columns=f"tot_score", inplace=True)
                except:
                    pass
                df_sim = df_sim.merge(
                    emdat_[(emdat_.value_a == a) & (emdat_.value_c == c)]
                    [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
                df_sim['tot_score'] = df_sim['tot_score'].fillna(0)
                df_sim['sim_senders'] = df_sim.apply(simulate_row_grouped_deterministic, axis=1)
                list_sims.append(df_sim)
            df_country_phil = pd.concat(list_sims)
            df_country_phil['sim_remittances'] = df_country_phil['sim_senders'] * fixed_remittance_phil
            df_country = pd.concat([df_country_ita, df_country_phil])
            remittance_per_period = df_country.groupby(['date', 'origin', 'destination'])['sim_remittances'].sum().reset_index()
            remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")
            remittance_per_period['error_squared'] = (remittance_per_period['remittances'] -
                                                      remittance_per_period['sim_remittances']) ** 2

            dict_params = {"nta" : param_nta,
                           "asy" : param_asy,
                           "stay" : param_stay,
                           "gdp" : param_gdp,
                           "a" : a,
                           "c" : c,
                           "rem_value" : fixed_remittance}
            results_run = [dict_params, remittance_per_period['error_squared'].mean()] + [f]
            results_list.append(results_run)

            df_country_ita = df_country_ita_.copy()
            df_country_phil = df_country_phil_.copy()

import pickle
with open('model_results_ita_phil_disasters.pkl', 'wb') as fi:
    pickle.dump(results_list, fi)

with open('model_results_ita_phil_disasters.pkl', 'rb') as fi:
    loaded_data = pickle.load(fi)


min_tuple_each_run = []
for f in tqdm(range(n_repetitions)):
    sub_data = [x for x in results_list if x[2] == f]
    flattened_data = [item[1] for item in sub_data]
    min_tuple_index = flattened_data.index(min(flattened_data))
    min_tuple = sub_data[min_tuple_index]
    min_tuple_each_run.append(min_tuple)

def r_squared_all_and_mean():
    countries_ita = ita_all_countries
    countries_phil = phil_all_countries

    global nta_dict
    # df country
    df_country_ita = df.query(f"""`origin` in {countries_ita} and `destination` == 'Italy'""")
    df_country_phil = df.query(f"""`origin` == 'Philippines' and `destination` in {countries_phil}""")
    df_country_ita = df_country_ita[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    df_country_phil = df_country_phil[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    # asy
    asy_df_ita = asy_df.query(f"""`destination` == 'Italy'""")
    asy_df_phil = asy_df.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.sort_values(['origin', 'date']).merge(asy_df_ita[["date", "asymmetry", "origin"]],
                                          on=["date", "origin"], how='left').ffill()
    df_country_phil = df_country_phil.sort_values(['origin', 'date']).merge(asy_df_phil[["date", "asymmetry", "destination"]],
                                            on=["date", "destination"], how='left').ffill()
    # growth rates
    growth_rates_ita = growth_rates.query(f"""`destination` == 'Italy'""")
    growth_rates_phil = growth_rates.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.merge(growth_rates_ita[["date", "yrly_growth_rate", "origin"]],
                                          on=["date", "origin"], how='left')
    df_country_phil = df_country_phil.merge(growth_rates_phil[["date", "yrly_growth_rate", "destination"]],
                                            on=["date", "destination"], how='left')

    df_country_ita['yrly_growth_rate'] = df_country_ita['yrly_growth_rate'].bfill()
    df_country_ita['yrly_growth_rate'] = df_country_ita['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country_ita = df_country_ita.merge(df_betas, on="yrly_growth_rate", how='left')
    df_country_phil['yrly_growth_rate'] = df_country_phil['yrly_growth_rate'].bfill()
    df_country_phil['yrly_growth_rate'] = df_country_phil['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country_phil = df_country_phil.merge(df_betas, on="yrly_growth_rate", how='left')

    ##gdp diff
    df_gdp_ita = df_gdp.query(f"""`destination` == 'Italy'""")
    df_gdp_phil = df_gdp.query(f"""`origin` == 'Philippines'""")
    df_country_ita = df_country_ita.merge(df_gdp_ita[["date", "gdp_diff_norm", "origin"]], on=["date", "origin"],
                                          how='left')
    df_country_ita['gdp_diff_norm'] = df_country_ita['gdp_diff_norm'].bfill()
    df_country_phil = df_country_phil.merge(df_gdp_phil[["date", "gdp_diff_norm", "destination"]],
                                            on=["date", "destination"],
                                            how='left')
    df_country_phil['gdp_diff_norm'] = df_country_phil['gdp_diff_norm'].bfill()

    ## nta
    df_nta_ita = df_nta.query(f"""`country` == 'Italy'""")[['age', 'nta']].fillna(0)
    df_nta_phil = df_nta[df_nta.country.isin(countries_phil)][['country', 'age', 'nta']].fillna(0)

    emdat_ = df_scores[df_scores.origin.isin(countries_ita + ["Philippines"])]
    dis_params['tot'] = [a, c]
    try:
        df_country_ita.drop(columns=f"tot_score", inplace=True)
    except:
        pass
    df_country_ita = df_country_ita.merge(
        emdat_[(emdat_.disaster == 'tot') & (emdat_.value_a == a) & (emdat_.value_c == c)]
        [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
    df_country_ita['tot_score'] = df_country_ita['tot_score'].fillna(0)

    df_country_ita['sim_senders'] = df_country_ita.apply(simulate_row_grouped_deterministic, axis=1)
    df_country_ita['sim_remittances'] = df_country_ita.sim_senders * fixed_remittance_ita

    list_sims = []
    for country in tqdm(
            list(set(df_nta_phil.country).intersection(set(phil_all_countries)))):
        df_sim = df_country_phil[df_country_phil.destination == country].copy()
        for ind, row in df_nta_phil[df_nta_phil.country == country].iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        dis_params['tot'] = [a, c]
        try:
            df_sim.drop(columns=f"tot_score", inplace=True)
        except:
            pass
        df_sim = df_sim.merge(
            emdat_[(emdat_.value_a == a) & (emdat_.value_c == c)]
            [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
        df_sim['tot_score'] = df_sim['tot_score'].fillna(0)
        df_sim['sim_senders'] = df_sim.apply(simulate_row_grouped_deterministic, axis=1)
        list_sims.append(df_sim)
    df_country_phil = pd.concat(list_sims)
    df_country_phil['sim_remittances'] = df_country_phil['sim_senders'] * fixed_remittance_phil
    df_country = pd.concat([df_country_ita, df_country_phil])
    remittance_per_period = df_country.groupby(['date', 'origin', 'destination'])['sim_remittances'].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")
    remittance_per_period['error_squared'] = (remittance_per_period['remittances'] -
                                              remittance_per_period['sim_remittances']) ** 2
    remittance_per_period['error'] = remittance_per_period['remittances'] - remittance_per_period['sim_remittances']
    SS_res = np.sum(np.square(remittance_per_period['error']))
    SS_tot = np.sum(np.square(remittance_per_period['remittances'] - np.mean(remittance_per_period['remittances'])))
    R_squared = 1 - (SS_res / SS_tot)
    print(f"R-squared: {round(R_squared, 3)}")
    df_mean = remittance_per_period[['origin', 'remittances', 'sim_remittances']].groupby(['origin']).mean().reset_index()
    df_mean['error'] = df_mean['remittances'] - df_mean['sim_remittances']
    SS_res = np.sum(np.square(df_mean['error']))
    SS_tot = np.sum(np.square(df_mean['remittances'] - np.mean(df_mean['remittances'])))
    R_squared = 1 - (SS_res / SS_tot)
    print(f"R-squared means: {round(R_squared, 3)}")

for min_tuple in min_tuple_each_run:
    best_params = min_tuple[0]
    print(best_params)
    print(f"Abs error: {min_tuple[1] / 1_000_000_000_000}")
    param_nta = best_params["nta"]
    param_stay = best_params["stay"]
    param_asy = best_params["asy"]
    param_gdp = best_params["gdp"]
    a = best_params["a"]
    c = best_params["c"]
    fixed_remittance = best_params["rem_value"]
    r_squared_all_and_mean()

#########################
a = 0.5
c = 0.5

param_nta = best_params["nta"]
param_stay = best_params["stay"]
param_asy = best_params["asy"]
param_gdp = best_params["gdp"]

check_initial_guess()