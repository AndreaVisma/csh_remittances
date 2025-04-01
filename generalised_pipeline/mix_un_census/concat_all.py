
import pandas as pd
import numpy as np
import re
from scipy.interpolate import CubicSpline
from tqdm import tqdm

latam = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\latam\\processed_latam_hosts_3obs.pkl")
euro = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\europe\\processed_european_hosts_3obs.pkl")
row = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\world\\processed_row_hosts_3obs.pkl")
##
df_us = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\us\\processed_usa.pkl")
df_us = df_us[df_us.date.isin(["2010-01-31", "2015-01-31", "2020-01-31"])]
df_us['mean_age'] = df_us['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
df_us["age_group"] = pd.cut(df_us.mean_age, bins = bins).map(dict(zip(bins, labels)))
df_us = df_us[['date', 'origin', 'destination', 'age_group', 'sex', 'n_people']]

#
ger = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\germany\\processed_germany.pkl")
ger = ger[ger.date.isin(["2010-01-31", "2015-01-31", "2020-01-31"])]
ger['mean_age'] = ger['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
ger["age_group"] = pd.cut(ger.mean_age, bins = bins).map(dict(zip(bins, labels)))
ger = ger[['date', 'origin', 'destination', 'age_group', 'sex', 'n_people']]

ita = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\italy\\processed_italy.pkl")
ita = ita[ita.date.isin(["2010-01-31", "2015-01-31", "2020-01-31"])]
ita['mean_age'] = ita['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
ita["age_group"] = pd.cut(ita.mean_age, bins = bins).map(dict(zip(bins, labels)))
ita = ita[['date', 'origin', 'destination', 'age_group', 'sex', 'n_people']]

##
df_all = pd.concat([latam, euro, row, df_us, ger, ita])
df_all.to_pickle("C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_3obs.pkl")
#
# # Preprocess date and sort
# df_all['date'] = pd.to_datetime(df_all['date'])
# df_all.sort_values('date', inplace=True)
#
# # Precompute mean age mapping
# age_mean_dict = {
#     ag: np.mean(list(map(int, re.findall(r'\d+', ag))))
#     for ag in df_all['age_group'].unique()
# }
#
# df_months = []
#
# # Process each destination country
# for dest_country in tqdm(df_all['destination'].unique()[2:], desc="Processing Destinations"):
#     df_dest = df_all[df_all['destination'] == dest_country]
#     if df_dest.empty:
#         continue
#
#     # Precompute time values
#     start_date = df_dest['date'].min()
#     end_date = df_dest['date'].max()
#     monthly_dates = pd.date_range(start_date, end_date, freq='M')
#     monthly_times = (monthly_dates - start_date).days
#     df_dest['time'] = (df_dest['date'] - start_date).dt.days
#
#     # Group by origin, age_group, sex and filter groups with sufficient data
#     grouped = df_dest.groupby(['origin', 'age_group', 'sex'])
#     dest_data = []
#
#     for (origin, age, sex), group in grouped:
#         if len(group) < 2:
#             continue  # Skip groups with insufficient data points
#
#         try:
#             cs = CubicSpline(group['time'], group['n_people'])
#             vals = cs(monthly_times)
#         except Exception:
#             continue
#
#         # Create DataFrame for valid group
#         dest_data.append(pd.DataFrame({
#             'date': monthly_dates,
#             'origin': origin,
#             'age_group': age,
#             'sex': sex,
#             'n_people': vals.round().astype(int),
#             'mean_age': age_mean_dict[age],
#             'destination': dest_country
#         }))
#
#     if dest_data:
#         df_months.extend(dest_data)
#
# # Concatenate all results at once
# df_months = pd.concat(df_months, ignore_index=True)
#
df_all['date'] = pd.to_datetime(df_all['date'])
df_all.sort_values('date', inplace = True)

list_df_months = []
pbar = tqdm(df_all.destination.unique()[2:])
for dest_country in pbar:
    pbar.set_description(f"Processing {dest_country} .. ")
    df_destination = df_all[df_all.destination == dest_country].copy()
    start_date, end_date = df_destination['date'].min(), df_destination['date'].max()
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
    monthly_times = (monthly_dates - start_date).days
    df_destination['time'] = (df_destination['date'] - df_destination['date'].min()).dt.days

    cols = ['date', 'origin', 'age_group', 'sex', 'n_people']
    list_dfs = []

    for country in df_destination.origin.unique():
        for age_group in df_destination.age_group.unique():
            for sex in df_destination.sex.unique():
                try:
                    data = [monthly_dates,
                            [country] * len(monthly_dates),
                            [age_group] * len(monthly_dates),
                            [sex] * len(monthly_dates)]
                    cs = CubicSpline(df_destination[(df_destination.origin == country) & (df_destination.sex == sex) & (df_destination.age_group == age_group)]['time'],
                                     df_destination[(df_destination.origin == country) & (df_destination.sex == sex) & (df_destination.age_group == age_group)]["n_people"])
                    vals = cs(monthly_times)
                    data.append(vals)
                    dict_country = dict(zip(cols, data))
                    country_df = pd.DataFrame(dict_country)
                    list_dfs.append(country_df)
                except:
                    country_df = pd.DataFrame([])
                    list_dfs.append(country_df)

    df_month = pd.concat(list_dfs)
    df_month['n_people'] = df_month['n_people'].astype(int)
    df_month['mean_age'] = df_month['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
    df_month['destination'] = dest_country
    list_df_months.append(df_month)

df_months = pd.concat(list_df_months)
df_months.to_pickle("C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_interpolated.pkl")
