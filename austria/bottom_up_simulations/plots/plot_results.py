import matplotlib.pyplot as plt
import numpy as np

def plot_all_results(df):

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df.obs_remittances, df.sim_remittances)
    ax.plot(np.linspace(0, 140, 140), np.linspace(0, 140, 140), color='red')
    plt.xlabel('observed remittances')
    plt.ylabel('simulated remittances')
    plt.grid()
    plt.show(block=True)

def plot_results_country(df, country):

    df_country = df[df.country == country].copy()
    n = np.linspace(2011, 2023, 13)
    fig, ax = plt.subplots(figsize=(9, 6))
    x = df_country.obs_remittances.to_list()
    y = df_country.sim_remittances.to_list()
    ax.scatter(x, y)
    ax.plot(np.linspace(0, max(x), 140), np.linspace(0, max(x), 140), color='red')
    plt.xlabel('observed remittances')
    plt.ylabel('simulated remittances')
    plt.grid()
    plt.title(f"Observed v. simulated remittances for {country}")
    for i, txt in enumerate(n):
        ax.annotate(str(txt), (x[i], y[i]))
    plt.show(block=True)