
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import CubicSpline

df = pd.read_csv("c:\\data\\migration\\italy\\estimated_stocks_new.csv")
df.loc[df['age_group'] == "Less than 5 years", "age_group"] = "0 to 5"
df['age_mean'] = df['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=21, end = 105, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(21)]
df["age_group"] = pd.cut(df.age_mean, bins = bins).map(dict(zip(bins, labels)))

df.rename(columns = dict(zip(["citizenship", "count"],["origin", "n_people"])), inplace = True)
df['destination'] = "Italy"

df['date'] = pd.to_datetime(df['year'], format = "%Y")
df.sort_values('date', inplace = True)

################ fix zimbabwe
# countries_to_fix = ["Zimbabwe", "United Kingdom"]
# for country in tqdm(countries_to_fix):
#     zimb = df[(df.origin == country)].sort_values("n_people", ascending = False)
#     ind_to_drop = zimb[zimb[["origin", "year", "sex", "age_group"]].duplicated()].index
#     df = df.drop(ind_to_drop, axis=0)
################

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
df_month['destination'] = "Italy"

df_month.to_pickle("C:\\Data\\migration\\bilateral_stocks\\italy\\processed_italy.pkl")

