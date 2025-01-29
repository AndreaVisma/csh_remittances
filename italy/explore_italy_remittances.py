
import pandas as pd
import numpy as np
import country_converter as coco
cc = coco.CountryConverter()
from utils import dict_names
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import seaborn as sns

file = "C:\\Data\\remittances\\italy\\rimesse.xlsx"

## inflation correction
inflation = pd.read_excel("C:\\Data\\economic\\annual_inflation_clean.xlsx").query("Country == 'Italy' & year >= 2005")
inflation.loc[len(inflation)] = ["Italy", 2024, 1.1]
inflation.rename(columns = {'hcpi' : 'rate'}, inplace = True)
inflation['hcpi'] = 100
for year in tqdm(inflation.year.unique()[1:]):
    inflation.loc[inflation.year == year, 'hcpi'] = (inflation.loc[inflation.year == year - 1, 'hcpi'].item() *
                                                     (1 + inflation.loc[inflation.year == year, 'rate'].item() / 100))
inflation['hcpi'] = inflation['hcpi'] / 100

## monthly 2005-2015

df_month = pd.read_excel(file, sheet_name="mensili_2005_2015", parse_dates=["data"])
df_month['country'] = cc.convert(df_month["codice_paese"], to='names', not_found=np.nan)
df_month["country"] = df_month["country"].map(dict_names)
df_month.dropna(inplace = True)
df_month["remittances"] = df_month["importo (milioni di euro)"] * 1_000_000
df_month.rename(columns = {"data" : "date"}, inplace = True)
df_month = df_month[["date", "country", "remittances"]].copy()
for year in tqdm(df_month.date.dt.year.unique()):
    df_month.loc[df_month.date.dt.year == year, 'remittances'] = (df_month.loc[df_month.date.dt.year == year, 'remittances']/
                                                                      inflation[inflation.year == year]['hcpi'].item())

## quarterly 2016-2024
df_q = pd.read_excel(file, sheet_name="trimestrali_paese").iloc[1:]
df_q["date"] = pd.to_datetime(df_q['anno'].astype(str)+'Q'+df_q['trimestre'].astype(str)) + pd.tseries.offsets.QuarterEnd()
df_q['country'] = cc.convert(df_q["codice_paese"], to='names', not_found=np.nan)
df_q["country"] = df_q["country"].map(dict_names)
df_q.dropna(inplace = True)
df_q["remittances"] = df_q["importo (milioni di euro)"] * 1_000_000
df_q = df_q[["date", "country", "remittances"]].copy()
for year in tqdm(df_q.date.dt.year.unique()):
    df_q.loc[df_q.date.dt.year == year, 'remittances'] = (df_q.loc[df_q.date.dt.year == year, 'remittances']/
                                                                      inflation[inflation.year == year]['hcpi'].item())

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
df.to_csv("C:\\Data\\remittances\\italy\\monthly_splined_remittances.csv", index = False)
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
fig.savefig("C:\\git-projects\\csh_remittances\\italy\\plots\\remittances\\flows\\total_monthly_flows.svg")
plt.show(block = True)

### plot biggest receivers
def plot_biggest_receivers_years(years = [2005, 2024]):
    miny, maxy = years[0], years[1]
    df_group_country = df[(df.date >= str(miny)) & (df.date <= str(maxy + 1))][["country", "remittances"]].groupby("country").sum().reset_index()
    df_group_country['pct'] = 100 * df_group_country["remittances"] / df_group_country["remittances"].sum()
    df_group_country.loc[df_group_country.pct < 2, "country"] = "other"
    df_group_country = df_group_country[["country", "remittances", "pct"]].groupby("country").sum().reset_index()
    df_sorted = df_group_country.sort_values('remittances', ascending=False)

    colors = sns.color_palette('pastel')
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.pie(df_sorted['remittances'],
            labels=df_sorted['country'],
            autopct='%1.1f%%',
            colors=colors)
    plt.title(f'Top Countries by Remittances Received from Italy {miny}-{maxy}')
    plt.axis('equal')  # Equal aspect ratio for a circular pie
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.show(block = True)
    fig.savefig(f"C:\\git-projects\\csh_remittances\\italy\\plots\\remittances\\share_senders\\biggest_senders_{miny}-{maxy}.svg")

years_couples = [[2005, 2024], [2005, 2010], [2010, 2017], [2017, 2024]]
for couple in tqdm(years_couples):
    plot_biggest_receivers_years(years = couple)

### plot country-specific flows
def plot_country_monthly_flows(country, show = True):
    df_country = df[df.country == country]
    df_country_q = df_q[df_q.country == country].copy()
    df_country_q["remittances"] = df_country_q["remittances"] / 3

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df_country["date"], df_country["remittances"])
    ax.scatter(df_country_q["date"], df_country_q["remittances"], color='red')
    plt.grid()
    plt.title(f"Remittances flow from Italy to {country}")
    plt.ylabel("Euros")
    # plt.ticklabel_format(axis='y', style='plain')
    fig.savefig(f"C:\\git-projects\\csh_remittances\\italy\\plots\\remittances\\flows\\total_monthly_flows_{country}.svg")
    if show:
        plt.show(block=True)

for country in tqdm(df.country.unique()):
    plot_country_monthly_flows(country, show = False)

plot_country_monthly_flows('Mexico', show = True)
plot_country_monthly_flows('China')
plot_country_monthly_flows('Bangladesh')
plot_country_monthly_flows('Pakistan')
