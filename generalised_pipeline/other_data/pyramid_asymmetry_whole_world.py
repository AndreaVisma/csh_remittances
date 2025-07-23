
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import re

diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_interpolated.pkl"
df = pd.read_pickle(diasporas_file)
df.loc[df.n_people < 0, 'n_people'] *= -1
df.date = pd.to_datetime(df.date)
df = df[df.date.dt.month == 1]
# df = df[df.n_people != 0 ]

df_all_factors = []
pbar = tqdm(df.destination.unique())

for country_dest in pbar:
    pbar.set_description(f"Processing {country_dest} ...")
    df_month = df[df.destination == country_dest]
    countries, age_factors, sex_factors, periods = [], [], [], []
    for country in df_month.origin.unique():
        for period in df_month[df_month.origin == country].date.unique():

            young_count = df_month[(df_month.date == period) & (df_month.origin == country)
                                   & (df_month.mean_age <= 25)].n_people.sum()
            parenting_count = df_month[(df_month.date == period) & (df_month.origin == country)
                                   & (df_month.mean_age > 25) & (df_month.mean_age <= 50)].n_people.sum()
            male_count = df_month[(df_month.date == period) & (df_month.origin == country)
                                   & (df_month.sex == 'male')].n_people.sum()
            female_count = df_month[(df_month.date == period) & (df_month.origin == country)
                                   & (df_month.sex == 'female')].n_people.sum()
            if young_count > 0:
                age_factor = max(0, (1 - (parenting_count / (3 * young_count))))
            else:
                age_factor = 0
            if male_count + female_count > 0:
                sex_factor = min(male_count, female_count) / (0.5 * (male_count + female_count))
            else:
                sex_factor = 0

            countries.append(country)
            periods.append(period)
            age_factors.append(age_factor)
            sex_factors.append(sex_factor)

    df_factors = pd.DataFrame({'origin' : countries, 'date' : periods,
                               'age_asy' : age_factors, 'sex_asy' : sex_factors})
    df_factors['asymmetry'] = df_factors.age_asy * df_factors.sex_asy
    df_factors['destination'] = country_dest
    df_all_factors.append(df_factors)

df_all_factors = pd.concat(df_all_factors)
df_all_factors.to_pickle("C:\\Data\\migration\\bilateral_stocks\\pyramid_asymmetry_beginning_of_the_year_NEW.pkl")
