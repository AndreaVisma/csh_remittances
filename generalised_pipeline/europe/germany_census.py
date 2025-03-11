
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from utils import dict_names
from scipy.interpolate import CubicSpline

ger_new = "C:\\Users\\Andrea Vismara\\Downloads\\ger_new\\12521-0003_en_flat.csv"

df = pd.read_csv(ger_new, sep = ';', encoding="UTF-8-SIG")
df.rename(columns = dict(zip(["2_variable_attribute_label", "3_variable_attribute_label", "4_variable_attribute_label", "value"],
                             ["age_group", "origin", "sex", "n_people"])), inplace = True)
df = df[["time", "origin", "age_group", "sex", "n_people"]]
df = df[df.sex != "Total"]
df['origin'] = df['origin'].map(dict_names)
df.dropna(inplace = True)

df['age'] = df['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
df["age_group"] = pd.cut(df.age, bins = bins).map(dict(zip(bins, labels)))
df['sex'] = df['sex'].map(dict(zip(['Male', "Female"], ["male", "female"])))

df = (df[['time', 'origin', 'age_group', 'sex', 'n_people']]
      .groupby(['time', 'origin', 'age_group', 'sex']).sum().reset_index())

df['date'] = pd.to_datetime(df['time'])
df.sort_values('date', inplace = True)

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
df_month['destination'] = "Germany"

df_month.to_pickle("C:\\Data\\migration\\bilateral_stocks\\germany\\processed_germany.pkl")

df_ = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\germany\\processed_germany.pkl")




