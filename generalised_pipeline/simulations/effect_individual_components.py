

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time

## pair of countries
origin, destination = "Germany", "Italy"

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_interpolated.pkl"
df = pd.read_pickle(diasporas_file)

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
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters.pkl").rename(columns = {'country' : 'origin'})

#########################################
#########################################
# Sample parameters
param_nta = 1
param_stay = -0.05
param_asy = -2.5
param_gdp = 10
dis_params = pd.read_excel("C:\\Data\\my_datasets\\disasters\\disasters_params.xlsx", sheet_name="Sheet2").dropna()
# dict_dis_par = dict(zip(['eq', 'dr', 'fl', 'st', 'tot'], [dis_params[x].tolist() for x in ['eq', 'dr', 'fl', 'st', 'tot']]))

fixed_remittance = 400  # Amount each sender sends

#####################################
#####################################
def sin_function(a,c,x):
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
            values = [sin_function(a,c,x) for x in np.linspace(0, 11, 12)]
            values = zero_values_before_first_positive_and_after_first_negative(values.copy())
        dict_dis_par[dis] = values
    return dict_dis_par

dict_dis_par = disaster_score_function(disasters = ['tot'], simple=True)

def compute_disasters_scores_all_countries(df, values_a, values_c):
    df_list = []
    for disaster in ['tot']:
        for a in values_a:
            for c in values_c:
                df_disaster = pd.DataFrame([])
                params = [sin_function(a, c, x) for x in np.linspace(0, 11, 12)]
                disaster_cols = [f"{disaster}_{i}" for i in range(12)]
                weights = np.array([params[i] for i in range(12)])
                impacts = df[disaster_cols].values.dot(weights)
                df_disaster['origin'] = df['origin']
                df_disaster['date'] = df['date']
                df_disaster["value_a"] = round(a,2)
                df_disaster["value_c"] = round(c,2)
                df_disaster["disaster"] = disaster
                df_disaster[f"{disaster}_score"] = impacts
                df_list.append(df_disaster)
    df_output = pd.concat(df_list)
    return df_output

df_scores = pd.read_pickle("C:\\Data\\my_datasets\\disaster_scores_only_tot.pkl")

def parse_age_group(age_group_str):
      """Helper function to parse age_group.
         This expects strings like "20-24". """
      lower, upper = map(int, age_group_str.split('-'))
      return lower, upper

def individual_effects_country_pair(origin, destination, df, disasters = False):
    global nta_dict
    # df country
    df_country = df.query(f"""`origin` == '{origin}' and  `destination` == '{destination}'""")
    df_country = df_country[[x for x in df.columns if x != 'sex']].groupby(['date', 'origin', 'age_group', 'mean_age','destination']).mean().reset_index()
    # asy
    asy_df_country = asy_df.query(f"""`origin` == '{origin}' and  `destination` == '{destination}'""")
    df_country = df_country.merge(asy_df_country[["date", "asymmetry"]], on="date", how='left').interpolate()
    # growth rates
    growth_rates_cr = growth_rates.query(f"""`origin` == '{origin}' and  `destination` == '{destination}'""")
    df_country = df_country.merge(growth_rates_cr[["date", "yrly_growth_rate"]], on="date", how='left')
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].interpolate()
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country = df_country.merge(df_betas, on="yrly_growth_rate", how='left')
    ##gdp diff
    df_gdp_cr = df_gdp.query(f"""`origin` == '{origin}' and  `destination` == '{destination}'""")
    df_country = df_country.merge(df_gdp_cr[["date", "gdp_diff_norm"]], on="date", how='left')
    df_country['gdp_diff_norm'] = df_country['gdp_diff_norm'].interpolate()
    ## nta
    df_nta_country = df_nta.query(f"""`country` == '{destination}'""")[['age', 'nta']].fillna(0)
    ## disasters
    df_country = df_country.merge(
        df_scores[(df_scores.value_a == 0.6) & (df_scores.value_c == 1)]
        [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
    df_country['tot_score'] = df_country['tot_score'].fillna(0)
    if not disasters:
        df_country['tot_score'] = 0

    # Create your base df_plot
    df_country = df_country[df_country.age_group == "40-44"]
    df_plot = df_country.copy()
    df_plot['nta'] = param_nta * nta_dict[40]
    df_plot['stay'] = param_stay * df_plot['beta_estimate'].apply(lambda x: np.random.exponential(scale=x, size=1_000).mean())
    df_plot['asymmetry'] = param_asy * df_plot["asymmetry"]
    df_plot['gdp'] = param_gdp * df_plot["gdp_diff_norm"]

    df_plot = df_plot[['date', 'asymmetry', 'stay', 'nta', 'gdp', 'tot_score']]
    df_plot = df_plot.groupby('date').mean()

    # Create subplots
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot 1: Disaster Scores
    df_plot[['asymmetry', 'stay', 'nta', 'gdp', 'tot_score']].plot(ax=ax)
    ax.set_title("Theta Scores for each component")
    ax.set_ylabel("Score Value")
    ax.grid(True)
    ax.legend(loc='best')

    # Tight layout and show
    plt.tight_layout()
    plt.show(block = True)


individual_effects_country_pair("Bangladesh", "Italy", df, disasters = True)