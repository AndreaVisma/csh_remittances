
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

## nta
nta = pd.read_excel("c:\\data\\economic\\nta\\NTA profiles.xlsx", sheet_name="italy").T
nta.columns = nta.iloc[0]
nta = nta.iloc[1:]
nta.reset_index(names='age', inplace = True)
nta = nta[['age', 'Support Ratio']].rename(columns = {'Support Ratio' : 'nta'})
nta.nta=(nta.nta-nta.nta.min())/(nta.nta.max()-nta.nta.min()) - 0.15
nta.loc[nta.nta <0, 'nta'] = 0

## disasters
df_nat = pd.read_csv("C:\\Data\\my_datasets\\weekly_remittances\\weekly_disasters.csv")
df_nat["week_start"] = pd.to_datetime(df_nat["week_start"])
df_nat["year"] = df_nat.week_start.dt.year
dfpop_country = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")
df_nat = df_nat.merge(dfpop_country, on = ['country', 'year'], how = 'left')
df_nat['total_affected'] = 100 * df_nat['total_affected'] / df_nat["population"]
df_nat = df_nat[["week_start", "total_affected", "total_damage", "country", "type"]]
df_nat_monthly = (
    df_nat.groupby(['country', pd.Grouper(key='week_start', freq='M')])
    .agg({'total_affected': 'sum', 'total_damage': 'sum'})
    .reset_index()
    .rename(columns={'week_start': 'date'}))
df_nat_monthly = df_nat_monthly[(df_nat_monthly.date.dt.year > 2008) & (df_nat_monthly.date.dt.year < 2024)]
dates = df_nat_monthly['date'].sort_values().unique().tolist()

## person 1 -> Nepal
x = 37
list_age = [x] * 12
for i in range(14):
    x +=1
    list_age = list_age + [x for j in range(12)]
y = 1
list_years = [y] * 12
for i in range(14):
    y +=1
    list_years = list_years + [y for j in range(12)]
list_country = ['Nepal'] * len(df_nat_monthly.date.unique())
list_fam_prob = [0.4] * len(df_nat_monthly.date.unique())

df_nepal = pd.DataFrame.from_dict({'country' : list_country, 'age' : list_age,
                                   'stay_years': list_years, 'prob_fam': list_fam_prob,
                                   'date': dates})

## person 2 -> Philippines
x = 22
list_age = [x] * 12
for i in range(14):
    x +=1
    list_age = list_age + [x for j in range(12)]
y = 18
list_years = [y] * 12
for i in range(14):
    y +=1
    list_years = list_years + [y for j in range(12)]
list_country = ['Philippines'] * len(df_nat_monthly.date.unique())
list_fam_prob = [0.7] * len(df_nat_monthly.date.unique())

df_philippines = pd.DataFrame.from_dict({'country' : list_country, 'age' : list_age,
                                   'stay_years': list_years, 'prob_fam': list_fam_prob,
                                   'date': dates})

## person 3 -> Mexico
x = 55
list_age = [x] * 12
for i in range(14):
    x +=1
    list_age = list_age + [x for j in range(12)]
y = 6
list_years = [y] * 12
for i in range(14):
    y +=1
    list_years = list_years + [y for j in range(12)]
list_country = ['Bangladesh'] * len(df_nat_monthly.date.unique())
list_fam_prob = [0.2] * len(df_nat_monthly.date.unique())

df_mexico = pd.DataFrame.from_dict({'country' : list_country, 'age' : list_age,
                                   'stay_years': list_years, 'prob_fam': list_fam_prob,
                                   'date': dates})

## merge
df = pd.concat([df_nepal, df_philippines, df_mexico])
df = df.merge(nta, on='age', how = 'left')
df = df.merge(df_nat_monthly[['country', 'date', 'total_affected']], on = ['country', 'date'], how = 'left').fillna(0)
df['total_affected'] /= 100
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['country', 'date'], inplace = True)
df.reset_index(drop = True, inplace = True)
# Create shifted columns using proper datetime handling
for shift in tqdm([1, 2, 3, 4]):
    g = df.groupby(['country'], group_keys=False)
    g =  g.apply(lambda x: x.set_index('date')['total_affected']
                 .shift(shift).reset_index(drop=True)).fillna(0)
    list_g = [x for i, row in g.iterrows() for x in row]
    df[f'ta_{shift}'] = list_g

#### PARAMETERS
p_length = 0.004
def stay_prob(p_length, stay):
    return 1 - np.exp(p_length * stay)
test = [stay_prob(p_length,x) for x in np.linspace(0, 100, 100)]
plt.plot(test)
plt.show(block = True)

p_fam = -0.3
test = [p_fam * x for x in np.linspace(0, 1, 100)]
plt.plot(test)
plt.show(block = True)

# disasters
d_boost_0 = 1
d_boost_1 = 2
d_boost_2 = 3
d_boost_3 = 2.5
d_boost_4 = 1.5

## simulate probability
df['prob'] = df.nta
#demographic_factors
df['prob'] = df['prob'] + [stay_prob(p_length,x) for x in df['stay_years']]
df['prob'] = df['prob'] + [p_fam * x for x in df['prob_fam']]

## disasters effect
df.loc[df['prob'] > 0, 'prob'] += df['total_affected'] * d_boost_0
df.loc[df['prob'] > 0, 'prob'] += df['ta_1'] * d_boost_1
df.loc[df['prob'] > 0, 'prob'] += df['ta_2'] * d_boost_2
df.loc[df['prob'] > 0, 'prob'] += df['ta_3'] * d_boost_3
df.loc[df['prob'] > 0, 'prob'] += df['ta_4'] * d_boost_4

df.loc[df['prob'] < 0, 'prob'] = 0
df.loc[df['prob'] > 1, 'prob'] = 1

##plot
dict_persons = dict(zip(df.country.unique().tolist(),
                        ['Bangladesh, 55\n6 years of stay',
                         'Nepal, 37\n1 year of stay',
                         'Philippines, 22\n18 years of stay']))
df['Person'] = df.country.map(dict_persons)

fig, ax = plt.subplots(figsize = (10, 7))
sns.lineplot(df, x = 'date', y = 'prob', hue = 'Person', ax = ax)
# plt.legend().remove()
plt.xlabel('Date')
plt.ylabel('Probability of sending remittances')
plt.grid()
fig.savefig("C:\\git-projects\\csh_remittances\\italy\\plots\\plots_for_paper\\simulation\\three_individuals_sim.svg", bbox_inches = 'tight')
plt.show(block = True)