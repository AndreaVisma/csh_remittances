import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel.xlsx")

def plot_pop_rem_country(country):

    df_country = df[(df.country == country) & (df.growth_rate_rem != np.inf)].fillna(0)
    disaster_years = df_country[df_country.nat_dist_dummy == 1].year.tolist()

    fig,ax = plt.subplots(figsize= (9,6))
    ax2 = ax.twinx()
    ax.plot(df_country.year, df_country['pop'], '-o', label = 'population (lhs)', color = 'red')
    ax2.plot(df_country.year, df_country['mln_euros'], '-o', label = 'remittances (rhs)', color = 'blue')
    plt.grid()
    for year in disaster_years:
        plt.axvline(x=year, color = 'orange', label = 'natural\ndisasters' if year == disaster_years[0] else '')
    ax.legend(loc=(0.02, 0.91))
    ax2.legend(loc = (0.02, 0.77))
    plt.title(f"population and remittances for {country}")
    plt.show(block = True)

    fig,ax = plt.subplots(figsize= (9,6))
    ax2 = ax.twinx()
    ax.plot(df_country.year, df_country['growth_rate_pop'], '-o', label = 'population (lhs)', color = 'red')
    ax2.plot(df_country.year, df_country['growth_rate_rem'], '-o', label = 'remittances (rhs)', color = 'blue')
    plt.grid()
    for year in disaster_years:
        plt.axvline(x=year, color = 'orange', label = 'natural\ndisasters' if year == disaster_years[0] else '')
    ax.legend(loc=(0.02, 0.91))
    ax2.legend(loc = (0.02, 0.77))
    plt.title(f"Growth rate of population and remittances for {country}")
    plt.show(block = True)

plot_pop_rem_country('Mexico')