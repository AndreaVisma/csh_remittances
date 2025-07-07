
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter

def simulate_population_yrs_stay(lambda_start = 3, poisson = True, years = 100, arrivals_rate = 0.05, exits_rate = 0.02, initial_population_size = 1_000, random_leavers = True):

    growth_rate = arrivals_rate - exits_rate

    if poisson:
        population = np.random.poisson(lambda_start, initial_population_size)
    else:
        population = np.random.exponential(lambda_start, initial_population_size)

    population_0 = population.copy()
    n_people = [len(population)]

    for y in tqdm(range(years)):
        pop_to_remove = len(population) * exits_rate
        pop_to_add = int(len(population) * arrivals_rate)
        pop_who_stays = int(len(population) - pop_to_remove)

        #make people leave
        if not random_leavers:
            population = np.sort(population)
            exp_values = np.array([1/(1 + np.exp(-0.15 * (x - 5))) for x in population])
            sum_exp_values = sum(exp_values)
            probabilities = np.array([x/sum_exp_values for x in exp_values])
            population = np.random.choice(population, size = pop_who_stays, replace = False, p = probabilities)
        else:
            population = np.random.choice(population, size=pop_who_stays, replace=False)
        #make people who stayed age
        population = population + 1
        #add new arrivals
        population = np.append(population, np.array([0] * pop_to_add))
        n_people.append(len(population))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.arange(- 0.5, max(population) + 1.5, 5)

    # Plot histogram
    axes[0].hist([population_0, population], bins=bins,
             weights=[np.ones(len(population_0)) / len(population_0), np.ones(len(population)) / len(population)],
             label=['Time 0', f'Time {years}'],
             alpha=0.7, edgecolor='black')

    # Labels and title
    axes[0].set_xlabel('Years of stay')
    axes[0].set_ylabel('Percentage of population')
    axes[0].set_title(f'Years of stay distribution\nArrivals rate: {100 * arrivals_rate}%; exits rate: {100 * exits_rate}%')
    axes[0].legend()
    axes[0].grid()
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))

    axes[1].plot(n_people, marker='o', linestyle='-', color='b', label='Total number of people')
    axes[1].set_xlabel('Years')
    axes[1].set_ylabel('Number of people')
    axes[1].set_title('Diaspora population evolution')
    axes[1].legend()
    axes[1].grid()

    fig.savefig(f"C:\\git-projects\\csh_remittances\\years_stay_simulation\\plots\\arrivals_{arrivals_rate}_exits_{exits_rate}_lambda_{lambda_start}_growth_{growth_rate}.png")
    plt.show(block = True)

    return growth_rate, population


###########
growth_rate, population = simulate_population_yrs_stay(lambda_start = 13, poisson = False, years = 10,
                                                       arrivals_rate = 0.07, exits_rate = 0.09,
                                                       initial_population_size = 1000, random_leavers=True)

def plot_population_leaving_probability():
    population = np.linspace(0,30, 61)
    exp_values = np.array([1 / (1 + np.exp(-0.15 * (x - 5))) for x in population])
    sum_exp_values = sum(exp_values)
    probabilities = np.array([x / sum_exp_values for x in exp_values])

    fig,ax = plt.subplots()
    ax.plot(population, probabilities)
    ax.set_title("Probability of staying given years of stay")
    ax.set_xlabel("Years of stay")
    ax.set_ylabel("probability of staying")
    plt.grid()
    plt.show(block = True)

plot_population_leaving_probability()

###########

lambdas = [2, 3, 5, 7]
arrivals = [0.05, 0.1, 0.15, 0.2]
exits = [0.03, 0.07, 0.12, 0.17]

import matplotlib as mpl
mpl.use("Agg")

for lambda_s in tqdm(lambdas):
    for arr in arrivals:
        for ex in exits:
            gr, pop = simulate_population_yrs_stay(lambda_start=lambda_s, years=10, arrivals_rate=arr, exits_rate=ex)
