
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize

def simulate_population_yrs_stay(lambda_start = 3, years = 10,
                                 growth_rate = 0.05,
                                 initial_population_size = 1_000, random_leavers = True):

    population = np.random.exponential(lambda_start, initial_population_size).astype(int)
    population_0 = population.copy()
    n_people = [len(population)]

    for y in tqdm(range(years)):
        new_people_baseline = int(len(population) * 0.05)
        pop_from_gr_rate = int(len(population) * growth_rate)
        pop_to_add = max(pop_from_gr_rate, new_people_baseline)
        if pop_from_gr_rate > new_people_baseline:
            pop_to_remove = len(population) * 0
        else:
            pop_to_remove = new_people_baseline - pop_from_gr_rate
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
    axes[0].set_title(f'Years of stay distribution\nGrowth rate: {100 * growth_rate}%')
    axes[0].legend()
    axes[0].grid()
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))

    axes[1].plot(n_people, marker='o', linestyle='-', color='b', label='Total number of people')
    axes[1].set_xlabel('Years')
    axes[1].set_ylabel('Number of people')
    axes[1].set_title('Diaspora population evolution')
    axes[1].legend()
    axes[1].grid()

    fig.savefig(f"C:\\git-projects\\csh_remittances\\years_stay_simulation\\plots\\lambda_{lambda_start}_growth_{growth_rate}.png")
    plt.show(block = True)

    return growth_rate, population

gr, pop = simulate_population_yrs_stay(lambda_start = 15, years = 15,
                                 growth_rate = -0.2,
                                 initial_population_size = 1_000, random_leavers = True)

def simulate_error_beta_gr_rate(lambda_start = 3, years = 100,
                                 growth_rate = 0.05,initial_population_size = 1_000,
                                random_leavers = True):

    population = np.random.exponential(lambda_start, initial_population_size).astype(int)
    population_0 = population.copy()
    n_people = [len(population)]

    for y in range(years):
        new_people_baseline = int(len(population) * 0.1)
        pop_from_gr_rate = int(len(population) * growth_rate)
        pop_to_add = max(pop_from_gr_rate, new_people_baseline)
        if pop_from_gr_rate > new_people_baseline:
            pop_to_remove = len(population) * 0
        else:
            pop_to_remove = new_people_baseline - pop_from_gr_rate
        pop_who_stays = int(len(population) - pop_to_remove)

        # make people leave
        if not random_leavers:
            population = np.sort(population)
            exp_values = np.array([1 / (1 + np.exp(-0.15 * (x - 5))) for x in population])
            sum_exp_values = sum(exp_values)
            probabilities = np.array([x / sum_exp_values for x in exp_values])
            population = np.random.choice(population, size=pop_who_stays, replace=False, p=probabilities)
        else:
            population = np.random.choice(population, size=pop_who_stays, replace=False)
        # make people who stayed age
        population = population + 1
        # add new arrivals
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

sum_es = simulate_error_beta_gr_rate(lambda_start = 8, years = 15,
                                 growth_rate = -0.2,
                                 initial_population_size = 1_000, random_leavers = True)

gr_rates = [round(x, 2) for x in np.linspace(-0.5, 0.5, 21)]
betas = np.linspace(1, 25, 40)
rates, beta_mins, errors, all_dfs = [], [], [], []

for gr_rate in tqdm(gr_rates):
    for beta in betas:
        for i in range(10):
            results_loc = []
            sum_es = simulate_error_beta_gr_rate(lambda_start=beta, years=10,
                                         growth_rate=gr_rate,
                                         initial_population_size=1_000, random_leavers=True)

            errors.append(sum_es)
            beta_mins.append(beta)
            rates.append(gr_rate)
    df_betas = pd.DataFrame({"growth_rate": rates,
                             "beta": beta_mins, "error_squared": errors})
    rates, beta_mins, errors = [], [], []
    all_dfs.append(df_betas)

df_betas = pd.concat(all_dfs)
df_betas_group = df_betas.groupby(["growth_rate", "beta"]).mean().reset_index()
df_betas.sort_values(["growth_rate", "error_squared"], inplace = True)

best_betas = df_betas_group.groupby("growth_rate")["error_squared"].min().tolist()
df_betas = df_betas_group[df_betas_group.error_squared.isin(best_betas)]

df_betas.to_pickle("C:\\Data\\migration\\bilateral_stocks\\betas.pkl")


fig,ax = plt.subplots()
plt.scatter(x = df_betas['growth_rate'], y = df_betas["beta"])
plt.xlabel("Growth rate")
plt.ylabel("Estimated beta")
plt.grid()
ax.xaxis.set_major_formatter(PercentFormatter(1))
plt.show(block = True)

