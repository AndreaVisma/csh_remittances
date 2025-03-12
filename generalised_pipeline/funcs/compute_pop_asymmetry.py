
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import re

def cspline_destination_origin(df):
    df['date'] = pd.to_datetime(df['date'])
    df_all = pd.DataFrame([])
    pbar = tqdm(df.destination.unique())
    for destination in tqdm(pbar):
        pbar.set_description(f"Processing {destination}")
        df_dest = df[df.destination == destination].copy()
        start_date, end_date = df_dest['date'].min(), df_dest['date'].max()
        monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
        monthly_times = (monthly_dates - start_date).days
        df_dest['time'] = (df_dest['date'] - df_dest['date'].min()).dt.days

        cols = ['date', 'origin', 'age_group', 'sex', 'n_people']
        df_month = pd.DataFrame()
        for country in (df_dest.origin.unique()):
            for age_group in df_dest.age_group.unique():
                for sex in df_dest.sex.unique():
                    data = [monthly_dates,
                            [country] * len(monthly_dates),
                            [age_group] * len(monthly_dates),
                            [sex] * len(monthly_dates)]
                    cs = CubicSpline(df_dest[(df_dest.origin == country) & (df_dest.sex == sex) & (df_dest.age_group == age_group)]['time'],
                                     df_dest[(df_dest.origin == country) & (df_dest.sex == sex) & (df_dest.age_group == age_group)]["n_people"])
                    vals = cs(monthly_times)
                    data.append(vals)
                    dict_country = dict(zip(cols, data))
                    country_df = pd.DataFrame(dict_country)
                    df_month = pd.concat([df_month, country_df])

        df_month['n_people'] = df_month['n_people'].astype(int)
        df_month['mean_age'] = df_month['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
        df_month['destination'] = destination
        df_all = pd.concat([df_all, df_month])
    return df_all

def compute_pop_asymmetry(df):

    df_all = pd.DataFrame([])
    pbar = tqdm(df.destination.unique())

    for destination in tqdm(pbar):
        pbar.set_description(f"Processing {destination}")
        df_month = df[df.destination == destination]
        origin, age_factors, sex_factors, periods = [], [], [], []
        for country in df_month.origin.unique():
            for period in df_month.date.unique():
                young_count = df_month[(df_month.date == period) & (df_month.origin == country)
                                       & (df_month.mean_age <= 25)].n_people.sum()
                parenting_count = df_month[(df_month.date == period) & (df_month.origin == country)
                                           & (df_month.mean_age > 25) & (df_month.mean_age <= 50)].n_people.sum()
                male_count = df_month[(df_month.date == period) & (df_month.origin == country)
                                      & (df_month.sex == 'male')].n_people.sum()
                female_count = df_month[(df_month.date == period) & (df_month.origin == country)
                                        & (df_month.sex == 'female')].n_people.sum()

                age_factor = max(0, (1 - (parenting_count / (3 * young_count))))
                sex_factor = min(male_count, female_count) / (0.5 * (male_count + female_count))

                origin.append(country)
                periods.append(period)
                age_factors.append(age_factor)
                sex_factors.append(sex_factor)
        df_factors = pd.DataFrame({'origin': origin, 'date': periods,
                                   'age_asy': age_factors, 'sex_asy': sex_factors})
        df_factors['destination'] = destination
        df_factors['asymmetry'] = df_factors.age_asy * df_factors.sex_asy
        df_all = pd.concat([df_all, df_factors])
    return df_all
