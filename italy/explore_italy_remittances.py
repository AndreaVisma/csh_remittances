
import pandas as pd
import numpy as np
import country_converter as coco
cc = coco.CountryConverter()
from utils import dict_names
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import CubicSpline

file = "C:\\Data\\remittances\\italy\\rimesse.xlsx"

## monthly 2005-2015

df_month = pd.read_excel(file, sheet_name="mensili_2005_2015", parse_dates=["data"])
df_month['country'] = cc.convert(df_month["codice_paese"], to='names', not_found=np.nan)
df_month["country"] = df_month["country"].map(dict_names)
df_month.dropna(inplace = True)
df_month["remittances"] = df_month["importo (milioni di euro)"] * 1_000_000
df_month.rename(columns = {"data" : "date"}, inplace = True)
df_month = df_month[["date", "country", "remittances"]].copy()

## quarterly 2016-2024
df_q = pd.read_excel(file, sheet_name="trimestrali_paese").iloc[1:]
df_q["date"] = pd.to_datetime(df_q['anno'].astype(str)+'Q'+df_q['trimestre'].astype(str)) + pd.tseries.offsets.QuarterEnd()
df_q['country'] = cc.convert(df_q["codice_paese"], to='names', not_found=np.nan)
df_q["country"] = df_q["country"].map(dict_names)
df_q.dropna(inplace = True)
df_q["remittances"] = df_q["importo (milioni di euro)"] * 1_000_000
df_q = df_q[["date", "country", "remittances"]].copy()

#monthly spline
start_date, end_date = df_q['date'].min(), df_q['date'].max()
monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
monthly_times = (monthly_dates - start_date).days
df_q['time'] = (df_q['date'] - df_q['date'].min()).dt.days

cols = ['date', 'country', 'remittances']
df_res_month = pd.DataFrame()

for country in tqdm(df_q.country.unique()):
    data = [monthly_dates, [country] * len(monthly_dates)]
    for col in cols[2:]:
        cs = CubicSpline(df_q[df_q.country == country]['time'],
                         df_q[df_q.country == country][col])
        vals = cs(monthly_times)
        data.append(vals)
    dict_country = dict(zip(cols, data))
    country_df = pd.DataFrame(dict_country)
    df_res_month = pd.concat([df_res_month, country_df])

df_res_month.loc[df_res_month.remittances < 0, "remittances"] = 0
df_res_month["remittances"] /= 3

#plot total monthly outflow
df = pd.concat([df_month, df_res_month])
df_group = df[["date", "remittances"]].groupby('date').sum().reset_index()
df_group["remittances"] /= 1_000_000
df_group_q = df_q[["date", "remittances"]].groupby('date').sum().reset_index()
df_group_q["remittances"] =  df_group_q["remittances"] / 3_000_000

fig, ax = plt.subplots(figsize = (12,8))
plt.plot(df_group["date"], df_group["remittances"])
plt.scatter(df_group_q["date"], df_group_q["remittances"], color = 'red')
plt.grid()
plt.title("Total remittances flow from Italy")
plt.ylabel("Million Euros")
plt.ticklabel_format(axis='y', style='plain')
fig.savefig("C:\\git-projects\\csh_remittances\\italy\\plots\\remittances\\total_monthly_flows.svg")
plt.show(block = True)