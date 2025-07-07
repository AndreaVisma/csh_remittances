import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fixed_vars = ['agent_id', 'country', 'sex']

df_h = pd.read_excel("c:\\data\\my_datasets\\explore_rem.xlsx")
df_h = df_h[df_h.remittances > 5000000]
df_h = df_h[df_h.year == 2019]

df_age = pd.read_pickle("c:\\data\\population\\austria\\simulated_migrants_populations_2010-2024.pkl")
df = df_age[fixed_vars + [str(x) for x in range(2010, 2026)]]
df_age.columns = fixed_vars + [str(x) for x in range(2010, 2026)]
df_age = pd.melt(df, id_vars=fixed_vars, value_vars=df.columns[3:],
             value_name='age', var_name='year')
df_age['year'] = df_age['year'].astype(int)

def plot_age_dist_country(country):
    df_country = df_age[df_age.country == country]
    sns.histplot(df_country, x='age', hue='sex', stat='percent')
    plt.grid()
    plt.title(f'Age distribution for {country}')
    plt.show(block=True)
plot_age_dist_country('Czechia')

fig, ax = plt.subplots()
sns.lineplot(df_h, x = 'quarter', y = 'probability', hue = 'country', ax = ax)
plt.grid()

sns.histplot(df_pol[df_pol.year == 2019], x = 'age', hue = 'sex', stat = 'percent')
plt.show(block = True)