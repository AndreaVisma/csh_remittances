import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from whittaker_eilers import WhittakerSmoother

df = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx")

## check deviations from the mean for all growth rate values
for country in tqdm(df.country.unique()):
    df.loc[df.country == country, 'dev_from_abs_mean'] = (df.loc[df.country == country, 'growth_rate_rem']-
                                                          df.loc[df.country == country, 'growth_rate_rem'].mean())
    df.loc[df.country == country, 'abs_mean'] = df.loc[df.country == country, 'growth_rate_rem'].mean()

def plot_distribution_affected(country):
    df_country = df[(df.country == country) & (df['total affected'] > 0)].copy()
    df_country['total affected'].hist(bins = 50)
    plt.show(block = True)
    return df_country['total affected'].mean()

def plot_country_rem_disasters(country, threshold):
    df_country = df[df.country == country].copy()
    quarter_means = df_country[['quarter', 'growth_rate_rem']].groupby('quarter').mean().to_dict()['growth_rate_rem']
    for quarter in quarter_means.keys():
        df_country.loc[df.quarter == quarter, 'variance_gr'] = (df_country.loc[df.quarter == quarter, 'growth_rate_rem']
        - quarter_means[quarter])
    plt.plot(df_country.date, df_country.growth_rate_rem)
    for date in df_country[df_country['total affected'] > threshold]['date']:
        plt.axvline(date, color = 'red')
    plt.grid()
    plt.show(block = True)
    return df_country

mean_country = plot_distribution_affected('Afghanistan')
df_country = plot_country_rem_disasters('Afghanistan', 100_000)

df_test = df[df.country == 'Syria'][['date', 'growth_rate_rem']].copy()

whittaker_smoother = WhittakerSmoother(
    lmbda=2, order=1, data_length=len(df_test))
df_test['growth_rate_rem_smooth'] = whittaker_smoother.smooth(df_test['growth_rate_rem'].to_list())

df_test.growth_rate_rem_smooth.plot()
df_test.growth_rate_rem.plot()
plt.grid()
plt.show(block = True)


