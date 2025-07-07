
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import re

file_us = "C:\\Data\\migration\\bilateral_stocks\\us\\processed_asia_latam.xlsx"
df = pd.read_excel(file_us)
df = df[~df.duplicated(['origin', 'age_group', 'sex', 'year'])]
df['date'] = pd.to_datetime(df['year'], format = "%Y")

start_date, end_date = df['date'].min(), df['date'].max()
monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
monthly_times = (monthly_dates - start_date).days
df['time'] = (df['date'] - df['date'].min()).dt.days

cols = ['date', 'origin', 'age_group', 'sex', 'n_people']
df_month = pd.DataFrame()

for country in tqdm(df.origin.unique()):
    for age_group in df.age_group.unique():
        for sex in df.sex.unique():
            data = [monthly_dates,
                    [country] * len(monthly_dates),
                    [age_group] * len(monthly_dates),
                    [sex] * len(monthly_dates)]
            cs = CubicSpline(df[(df.origin == country) & (df.sex == sex) & (df.age_group == age_group)]['time'],
                             df[(df.origin == country) & (df.sex == sex) & (df.age_group == age_group)]["n_people"])
            vals = cs(monthly_times)
            data.append(vals)
            dict_country = dict(zip(cols, data))
            country_df = pd.DataFrame(dict_country)
            df_month = pd.concat([df_month, country_df])

df_month['n_people'] = df_month['n_people'].astype(int)
df_month['mean_age'] = df_month['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
df_month['destination'] = "USA"
df_month.to_pickle("C:\\Data\\migration\\bilateral_stocks\\us\\processed_usa.pkl")

countries, age_factors, sex_factors, periods = [], [], [], []

for country in tqdm(df_month.origin.unique()):
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

        countries.append(country)
        periods.append(period)
        age_factors.append(age_factor)
        sex_factors.append(sex_factor)

df_factors = pd.DataFrame({'origin' : countries, 'date' : periods,
                           'age_asy' : age_factors, 'sex_asy' : sex_factors})
df_factors['asymmetry'] = df_factors.age_asy * df_factors.sex_asy
df_factors.to_csv("C:\\Data\\migration\\bilateral_stocks\\us\\pyramid_asymmetry_us.csv", index = False)

##### plot
import matplotlib.pyplot as plt
df_factors_gr = df_factors.groupby(['origin']).mean().reset_index()
df_factors_gr['asymmetry'].hist()
plt.title('Asymmetry pyramid')
plt.show(block = True)

df_factors_gr['age_asy'].hist()
plt.title('Age asymmetry pyramid')
plt.show(block = True)

df_factors_gr['sex_asy'].hist()
plt.title('Sex asymmetry pyramid')
plt.show(block = True)

