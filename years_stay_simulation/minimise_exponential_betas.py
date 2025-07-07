
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize

def simulate_population_yrs_stay(lambda_start = 3, poisson = False, years = 10,
                                 arrivals_rate = 0.05, exits_rate = 0.02,
                                 initial_population_size = 1_000, random_leavers = True):

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

def simulate_error_beta(lambda_start = 3, poisson = True, years = 100,
                                 arrivals_rate = 0.05, exits_rate = 0.02,
                                 initial_population_size = 1_000, random_leavers = True,
                        disable_progress = False):

    growth_rate = arrivals_rate - exits_rate

    if poisson:
        population = np.random.poisson(lambda_start, initial_population_size).astype(int)
    else:
        population = np.random.exponential(lambda_start, initial_population_size).astype(int)

    population_0 = population.copy()
    n_people = [len(population)]

    for y in tqdm(range(years), disable = disable_progress):
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

    #### bin population
    unique_0, counts_0 = np.unique(population_0, return_counts=True)
    counts_0 = 100 * counts_0 / sum(counts_0)
    df_0 = pd.DataFrame({'yrs_stay' : unique_0, 'pct_people_0' : counts_0})

    unique_1, counts_1 = np.unique(population, return_counts=True)
    counts_1 = 100 * counts_1 / sum(counts_1)
    df_1 = pd.DataFrame({'yrs_stay' : unique_1, 'pct_people_1' : counts_1})

    df = df_0.merge(df_1, on = 'yrs_stay', how = 'outer')
    df.fillna(0, inplace = True)
    df['error_squared'] = np.abs(df['pct_people_0'] - df['pct_people_1']) ** 2

    return df['error_squared'].sum()


###########
sum_es = simulate_error_beta(lambda_start = 15, poisson = False, years = 10,
                                               arrivals_rate = 0.1, exits_rate = 0.05,
                                               initial_population_size = 1000, random_leavers=False)

##############################

arrivals = np.linspace(0, 0.5, 5)
exits = np.linspace(0, 0.5, 5)

arr_rates, ex_rates, beta_mins, errors = [], [], [], []

for arr_rate in tqdm(arrivals):
    for leave_rate in tqdm(exits):
        betas = np.linspace(4, 30, 15)
        results = []
        for beta in betas:
            for i in range(5):
                results_loc = []
                sum_es = simulate_error_beta(lambda_start=beta, poisson=False, years=10,
                                             arrivals_rate=arr_rate, exits_rate=leave_rate,
                                             initial_population_size=10_000, random_leavers=False,
                                             disable_progress=True)
                results_loc.append(sum_es)
            results.append(np.mean(results_loc))
        errors.append(results[results.index(min(results))])
        beta_mins.append(betas[results.index(min(results))])
        arr_rates.append(arr_rate)
        ex_rates.append(leave_rate)

df_betas = pd.DataFrame({"arrival_rate" : arr_rates, "exit_rate" : ex_rates,
                         "est_beta" : beta_mins, "error_squared" : errors})
df_betas['growth_rate'] = df_betas["arrival_rate"] - df_betas["exit_rate"]
df_betas.sort_values("growth_rate", inplace = True)

fig,ax = plt.subplots()
plt.scatter(x = df_betas['growth_rate'], y = df_betas["est_beta"])
plt.xlabel("Growth rate")
plt.ylabel("Estimated beta")
plt.grid()
ax.xaxis.set_major_formatter(PercentFormatter(1))
plt.show(block = True)

df_group = df_betas[['growth_rate', 'est_beta']].groupby('growth_rate').mean().reset_index()
# df_group = df_group[(df_group.growth_rate >= -0.2) & (df_group.growth_rate <= 0.2)]

import seaborn as sns
sns.regplot(data=df_group, x="growth_rate", y="est_beta", order = 2)
plt.grid(True)
plt.show(block = True)

growth_rate, population = simulate_population_yrs_stay(lambda_start=20,
                             arrivals_rate=0.025,
                             exits_rate=0.025,
                             random_leavers = True)