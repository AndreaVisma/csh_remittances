"""
Script: simulate_population.py
Author: Andrea Vismara
Date: 12/11/2024
Description: simulate all the diaspora populations over the years
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_all = pd.read_pickle("c:\\data\\population\\austria\\population_quarterly_merged.pkl")
df_ita = df_all[df_all.country == "Italy"].copy()
df_ita.sort_values('age', inplace = True)

age = df_ita['age'].values

age_param = -0.006
age_param_2 = 0.08
h_param = 40

def probability_func(age):
    exponent = (age_param * np.power(age - h_param, 2) +
            age_param_2 * age)
    base_prob = np.exp(exponent) / (1 + np.exp(exponent))
    df_ita['probability'] = base_prob

    fig,ax = plt.subplots()
    ax.plot(df_ita.age, df_ita.probability)
    plt.grid()
    plt.show(block=True)

probability_func(age)