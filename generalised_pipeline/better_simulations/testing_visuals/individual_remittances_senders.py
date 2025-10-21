


import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time
import itertools
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from random import sample, uniform
import random
from pandas.tseries.offsets import MonthEnd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from utils import zero_values_before_first_positive_and_after_first_negative, dict_names

param_stay = 0

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors_2207_TRAIN.pkl"
complete_diaspora_df = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\complete_stock_to_plot.pkl")
df = pd.read_pickle(diasporas_file)
df = df.dropna()
df['year'] = df.date.dt.year
# df = df[(df.year == 2015) & (df.date.dt.month == 12)]

## income groups
income_class = pd.read_excel("C:\\Data\\economic\\income_classification_countries_wb.xlsx")
income_class['country'] = income_class.country.map(dict_names)
income_class = income_class[["country", "group", "Region"]]

##df_gdp
df_gdp = pd.read_pickle("c:\\data\\economic\\gdp\\annual_gdp_per_capita_splined.pkl")
df = df.merge(df_gdp, on=['destination', 'date'], how='left')

## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags_NEW.pkl")

####parameters

params = [2.323125219174453, -8.90268722949454, 9.158294373208719,
            0.1788188275520765, 0.21924650014050806,-0.7500114294211869,
            -0.04579122940366171, 0.13737550869522205]
######## functions

param_nta, param_asy, param_gdp, height, shape, shift, constant, rem_pct = params

######## functions
def calculate_tot_score(emdat_ita, height, shape, shift):
    global dict_scores
    dict_scores = dict(zip([x for x in range(12)],
                           zero_values_before_first_positive_and_after_first_negative(
                               [height + shape * np.sin((np.pi / 6) * (x+shift)) for x in range(1, 13)])))
    for x in range(12):
        emdat_ita[f"tot_{x}"] = emdat_ita[[f"eq_{x}",f"dr_{x}",f"fl_{x}",f"st_{x}"]].sum(axis =1) * dict_scores[x]
    emdat_ita["tot_score"] = emdat_ita[[x for x in emdat_ita.columns if "tot" in x]].sum(axis =1)
    return emdat_ita[['date', 'origin', 'tot_score']]

def calculate_tot_score_specific(emdat_ita, height, shape, shift, disaster):
    global dict_scores
    dict_scores = dict(zip([x for x in range(12)],
                           zero_values_before_first_positive_and_after_first_negative(
                               [height + shape * np.sin((np.pi / 6) * (x+shift)) for x in range(1, 13)])))
    for x in range(12):
        emdat_ita[f"tot_{x}"] = emdat_ita[[f"eq_{x}",f"dr_{x}",f"fl_{x}",f"st_{x}"]].sum(axis =1) * dict_scores[x]
    disasters_dict = dict(zip(["Earthquake", "Flood", "Storm", "Drought"], ["eq", "fl", "st", "dr"]))
    dis_name = disasters_dict[disaster]

    emdat_ita[f"{dis_name}_score"] = emdat_ita[[x for x in emdat_ita.columns if dis_name in x]].sum(axis =1)
    return emdat_ita[['date', 'origin', f"{dis_name}_score"]]

def simulate_remittance_probability(df_countries, height, shape, shift, rem_pct, disasters = True, disaster_impact = 0.01):

    if disasters:
        df_countries['tot_score'] = disaster_impact
    else:
        df_countries['tot_score'] = 0

    # df_countries['rem_amount'] = rem_pct * df_countries['gdp'] / 12

    df_countries['theta'] = constant + (param_nta * (df_countries['nta'])) \
                            + (param_asy * df_countries['asymmetry']) + (param_gdp * df_countries['gdp_diff_norm']) \
                            + (df_countries['tot_score'])
    df_countries['probability'] = 1 / (1 + np.exp(-df_countries["theta"]))
    df_countries.loc[df_countries.nta <= 0.01, 'probability'] = 0
    return df_countries

df_prob = simulate_remittance_probability(df, height, shape, shift, rem_pct, disasters = False)
df_prob["senders"] = np.round(df_prob["n_people"] * df_prob["probability"])
df_prob["senders"] = df_prob["senders"].astype(int)


######
## sex analysis
df_prob_tot = df_prob[["origin", "destination", "age_group", "senders", "n_people"]].groupby(["origin", "destination", "age_group"]).sum().reset_index()
complete_diaspora_tot = (complete_diaspora_df[["origin", "destination", "age_group", "sex", "n_people"]]
                         .groupby(["origin", "destination", "age_group", "sex"]).sum().reset_index())
df_comp_merged = df_prob_tot.merge(complete_diaspora_tot, on = ["origin", "destination", "age_group"], how = "left")
df_comp_merged["sex_ratio"] = df_comp_merged["n_people_y"] / df_comp_merged["n_people_x"]
df_comp_merged["senders_sex"] = df_comp_merged["senders"] * df_comp_merged["sex_ratio"]
df_comp_merged = df_comp_merged.merge(income_class, left_on = 'origin', right_on = 'country', how = 'left')
df_comp_merged.dropna(inplace = True)
df_comp_merged["senders_sex"] = df_comp_merged["senders_sex"].astype(int)
df_comp_merged = df_comp_merged[~df_comp_merged.age_group.isin(["0-4", "5-9", "10-14"])]

def get_age_midpoint(age_group):
    if isinstance(age_group, str) and '-' in age_group:
        try:
            start, end = age_group.split('-')
            return (int(start) + int(end)) / 2
        except:
            return None
    else:
        return None

# Apply the mapping
df_comp_merged['age_midpoint'] = df_comp_merged['age_group'].apply(get_age_midpoint)

df_comp_merged["macro_group"] = "0-20"
df_comp_merged.loc[df_comp_merged.age_midpoint > 20, "macro_group"] = "20-39"
df_comp_merged.loc[df_comp_merged.age_midpoint > 40, "macro_group"] = "40-59"
df_comp_merged.loc[df_comp_merged.age_midpoint > 60, "macro_group"] = "60-99"


tot_male = df_comp_merged.loc[df_comp_merged.sex == "male", "n_people_y"].sum()
tot_female = df_comp_merged.loc[df_comp_merged.sex == "female", "n_people_y"].sum()
tot_migrants = df_comp_merged.n_people_y.sum()

tot_male_senders = df_comp_merged.loc[df_comp_merged.sex == "male", "senders_sex"].sum()
tot_female_senders = df_comp_merged.loc[df_comp_merged.sex == "female", "senders_sex"].sum()
tot_senders = df_comp_merged.senders_sex.sum()

pct_male_migrants = round(100 * tot_male / tot_migrants, 2)
pct_male_senders = round(100 * tot_male_senders / tot_senders, 2)

print(f"Pct senders who are male: {pct_male_senders}%")
print(f"Pct migrants who are male: {pct_male_migrants}%")

##
sender_stats = df_comp_merged.groupby(['sex', 'age_group', 'group']).agg({
    'n_people_y': 'sum',  # Total migrants
    'senders_sex': 'sum'  # Total senders
}).reset_index()

sex_group = df_comp_merged.groupby(['sex', 'group']).agg({
    'n_people_y': 'sum',  # Total migrants
    'senders_sex': 'sum'  # Total senders
}).reset_index()
for group in sex_group["group"].unique():
    tot_send = sex_group.loc[sex_group["group"] == group, "senders_sex"].sum()
    tot_mig = sex_group.loc[sex_group["group"] == group, "n_people_y"].sum()
    sex_group.loc[sex_group["group"] == group, "senders_sex"] /= tot_send
    sex_group.loc[sex_group["group"] == group, "n_people_y"] /= tot_mig

# age and sex
age_sex = df_comp_merged.groupby(['age_group', 'sex']).agg({
    'n_people_y': 'sum',  # Total migrants
    'senders_sex': 'sum'  # Total senders
}).reset_index()
for group in age_sex["age_group"].unique():
    tot_send = age_sex.loc[age_sex["age_group"] == group, "senders_sex"].sum()
    tot_mig = age_sex.loc[age_sex["age_group"] == group, "n_people_y"].sum()
    age_sex.loc[age_sex["age_group"] == group, "senders_sex"] /= tot_send
    age_sex.loc[age_sex["age_group"] == group, "n_people_y"] /= tot_mig

## age_group
age_group = df_comp_merged.groupby(['age_group', 'group']).agg({
    'n_people_y': 'sum',  # Total migrants
    'senders_sex': 'sum'  # Total senders
}).reset_index()
for group in age_group["group"].unique():
    tot_send = age_group.loc[age_group["group"] == group, "senders_sex"].sum()
    tot_mig = age_group.loc[age_group["group"] == group, "n_people_y"].sum()
    age_group.loc[age_group["group"] == group, "senders_sex"] /= tot_send
    age_group.loc[age_group["group"] == group, "n_people_y"] /= tot_mig

###### average age sex income
# 1. Calculate average age of senders by sex and income group
average_age_senders = df_comp_merged.groupby(['group']).apply(
    lambda x: np.average(x['age_midpoint'], weights=x['senders_sex'])
).reset_index(name='weighted_avg_age')

average_age = np.average(df_comp_merged['age_midpoint'], weights=df_comp_merged['senders_sex'])
print(f"Average age global: {average_age}")
print("=== AVERAGE AGE OF SENDERS ===")
print(average_age_senders)

# macro age
macro_avg_senders = df_comp_merged.groupby(['macro_group']).apply(
    lambda x: np.sum(x['senders_sex']) / tot_senders
).reset_index()

macro_age_group = df_comp_merged.groupby(['macro_group', 'group']).agg({
    'n_people_y': 'sum',  # Total migrants
    'senders_sex': 'sum'  # Total senders
}).reset_index()
for group in macro_age_group["group"].unique():
    tot_send = macro_age_group.loc[macro_age_group["group"] == group, "senders_sex"].sum()
    tot_mig = macro_age_group.loc[macro_age_group["group"] == group, "n_people_y"].sum()
    macro_age_group.loc[age_group["group"] == group, "senders_sex"] /= tot_send
    macro_age_group.loc[age_group["group"] == group, "n_people_y"] /= tot_mig

macro_age_sex = df_comp_merged.groupby(['macro_group', 'sex']).agg({
    'n_people_y': 'sum',  # Total migrants
    'senders_sex': 'sum'  # Total senders
}).reset_index()
macro_age_sex["pct_total"] = macro_age_sex["senders_sex"] / tot_senders
#####
# deep seek suggestions
import seaborn as sns
# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Calculate sender percentages by different categories
df_comp_merged['is_sender'] = df_comp_merged['senders_sex'] > 0

# 1. Statistics: Percentage of migrants who are senders by sex, age_group, and income group
sender_stats = df_comp_merged.groupby(['sex', 'age_group', 'group']).agg({
    'n_people_y': 'sum',  # Total migrants
    'senders_sex': 'sum'  # Total senders
}).reset_index()

sender_stats['pct_senders'] = (sender_stats['senders_sex'] / sender_stats['n_people_y']) * 100
sender_stats['pct_senders'] = sender_stats['pct_senders'].round(2)

# 2. Statistics: Percentage distribution of senders across categories
total_senders = sender_stats['senders_sex'].sum()
sender_stats['pct_of_all_senders'] = (sender_stats['senders_sex'] / total_senders) * 100
sender_stats['pct_of_all_senders'] = sender_stats['pct_of_all_senders'].round(2)

print("=== STATISTICS: PERCENTAGE OF MIGRANTS WHO ARE SENDERS ===")
print(sender_stats[['sex', 'age_group', 'group', 'n_people_y', 'senders_sex', 'pct_senders']].sort_values('pct_senders', ascending=False))

print("\n=== STATISTICS: PERCENTAGE DISTRIBUTION OF ALL SENDERS ===")
print(sender_stats[['sex', 'age_group', 'group', 'senders_sex', 'pct_of_all_senders']].sort_values('pct_of_all_senders', ascending=False))

# 3. Create visualizations

# Figure 1: Percentage of migrants who are senders (heatmap style)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Heatmap for males
male_data = sender_stats[sender_stats['sex'] == 'male'].pivot_table(
    index='age_group', columns='group', values='pct_senders', aggfunc='mean'
)
sns.heatmap(male_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0])
axes[0].set_title('Percentage of Male Migrants Who Are Senders\nby Age Group and Income Group', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Income Group')
axes[0].set_ylabel('Age Group')

# Heatmap for females
female_data = sender_stats[sender_stats['sex'] == 'female'].pivot_table(
    index='age_group', columns='group', values='pct_senders', aggfunc='mean'
)
sns.heatmap(female_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1])
axes[1].set_title('Percentage of Female Migrants Who Are Senders\nby Age Group and Income Group', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Income Group')
axes[1].set_ylabel('Age Group')

plt.tight_layout()
plt.show(block = True)

# Figure 2: Stacked bar chart - Distribution of senders across categories
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Stacked bar for sender distribution
pivot_senders = sender_stats.pivot_table(
    index=['age_group', 'group'],
    columns='sex',
    values='pct_of_all_senders',
    aggfunc='sum'
).reset_index()

pivot_senders.plot(
    x=['age_group', 'group'],
    kind='bar',
    stacked=True,
    ax=axes[0],
    color=['lightcoral', 'lightblue']
)
axes[0].set_title('Percentage Distribution of All Senders\nby Age Group, Income Group and Sex', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age Group - Income Group')
axes[0].set_ylabel('Percentage of All Senders (%)')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(title='Sex')

# Figure 3: Comparison of sender rates by sex and income group
sns.catplot(
    data=sender_stats,
    x='age_group',
    y='pct_senders',
    hue='sex',
    col='group',
    kind='bar',
    height=5,
    aspect=1.2,
    palette={'male': 'lightblue', 'female': 'lightcoral'}
)
plt.suptitle('Percentage of Migrants Who Are Senders by Age Group, Sex and Income Group',
             y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show(block = True)

# Figure 4: Detailed comparison plot
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Top left: Sender rates by age group and sex
sns.barplot(data=sender_stats, x='age_group', y='pct_senders', hue='sex', ax=axes[0,0])
axes[0,0].set_title('Sender Rates by Age Group and Sex', fontweight='bold')
axes[0,0].set_ylabel('Percentage of Migrants Who Are Senders (%)')

# Top right: Sender rates by income group and sex
sns.barplot(data=sender_stats, x='group', y='pct_senders', hue='sex', ax=axes[0,1])
axes[0,1].set_title('Sender Rates by Income Group and Sex', fontweight='bold')
axes[0,1].set_ylabel('Percentage of Migrants Who Are Senders (%)')

# Bottom left: Distribution of senders by age group
age_sender_dist = sender_stats.groupby(['age_group', 'sex'])['pct_of_all_senders'].sum().reset_index()
sns.barplot(data=age_sender_dist, x='age_group', y='pct_of_all_senders', hue='sex', ax=axes[1,0])
axes[1,0].set_title('Distribution of All Senders by Age Group and Sex', fontweight='bold')
axes[1,0].set_ylabel('Percentage of All Senders (%)')

# Bottom right: Distribution of senders by income group
income_sender_dist = sender_stats.groupby(['group', 'sex'])['pct_of_all_senders'].sum().reset_index()
sns.barplot(data=income_sender_dist, x='group', y='pct_of_all_senders', hue='sex', ax=axes[1,1])
axes[1,1].set_title('Distribution of All Senders by Income Group and Sex', fontweight='bold')
axes[1,1].set_ylabel('Percentage of All Senders (%)')

plt.tight_layout()
plt.show(block = True)

# 4. Create summary tables for easy reading
print("\n=== SUMMARY: TOP 10 COMBINATIONS WITH HIGHEST SENDER RATES ===")
top_sender_rates = sender_stats.nlargest(10, 'pct_senders')[['sex', 'age_group', 'group', 'pct_senders', 'n_people_y']]
print(top_sender_rates)

print("\n=== SUMMARY: TOP 10 COMBINATIONS CONTRIBUTING MOST SENDERS ===")
top_sender_contributors = sender_stats.nlargest(10, 'pct_of_all_senders')[['sex', 'age_group', 'group', 'pct_of_all_senders', 'senders_sex']]
print(top_sender_contributors)

# 5. Additional detailed breakdown
print("\n=== DETAILED BREAKDOWN BY INCOME GROUP ===")
income_breakdown = sender_stats.groupby(['group', 'sex']).agg({
    'n_people_y': 'sum',
    'senders_sex': 'sum'
}).reset_index()
income_breakdown['pct_senders'] = (income_breakdown['senders_sex'] / income_breakdown['n_people_y'] * 100).round(2)
income_breakdown['pct_of_all_senders'] = (income_breakdown['senders_sex'] / total_senders * 100).round(2)
print(income_breakdown)


###################
###################


########
# senders per age group
tot_senders = df_prob.senders.sum()
senders_per_age = df_prob[["age_group", "senders"]].groupby("age_group").sum().reset_index()
senders_per_age["senders"] = 100 * senders_per_age["senders"] / tot_senders
print(senders_per_age)

## age per country group
df_prob_merge = df_prob.merge(income_class, left_on = "origin", right_on = "country", how = "left")
print(df_prob.isna().sum())
print(df_prob_merge.isna().sum())
df_prob_merge.dropna(inplace = True)

senders_age_income = df_prob_merge[["age_group", "group", "senders", "n_people"]].groupby(["age_group", "group"]).sum().reset_index()
senders_age_income = senders_age_income[~senders_age_income.age_group.isin(["0-4", "5-9", "10-14"])]

# Percentage of senders within each (age_group, group)
senders_age_income["share_senders"] = 0
for group in senders_age_income.group.unique():
    senders_age_income.loc[senders_age_income["group"] == group, "share_senders"] = (
            100 * senders_age_income["senders"] / senders_age_income.loc[senders_age_income["group"] == group, "senders"].sum())

# Total remittances (senders) by age group
age_distribution = senders_age_income.groupby("age_group")["senders"].sum().reset_index()
age_distribution["pct_world"] = 100 * age_distribution["senders"] / age_distribution["senders"].sum()

# Also include breakdown by income group
group_distribution = senders_age_income.groupby(["age_group", "group"])["senders"].sum().reset_index()
group_distribution["pct_within_age"] = group_distribution.groupby("age_group")["senders"].transform(lambda x: 100 * x / x.sum())


#####
# plot

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(8,5))
sns.barplot(data=age_distribution, x="age_group", y="pct_world", color="steelblue")
plt.title("Share of global remittance senders by age group")
plt.ylabel("Percentage of total senders (%)")
plt.xlabel("Age group")
plt.show(block = True)

####
pivot = group_distribution.pivot(index="age_group", columns="group", values="pct_within_age").fillna(0)
pivot.plot(kind="bar", stacked=True, figsize=(9,6), colormap="viridis")

plt.title("Composition of remittance senders by age and income group of origin")
plt.ylabel("Percentage within age group (%)")
plt.xlabel("Age group")
plt.legend(title="Income group of origin", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show(block = True)

#####
plt.figure(figsize=(9,6))
sns.barplot(data=senders_age_income, x="age_group", y="share_senders", hue="group", palette="mako")
plt.title("Probability of sending remittances by age and origin income group")
plt.ylabel("Share of migrants who send remittances")
plt.xlabel("Age group")
plt.legend(title="Income group of origin")
plt.tight_layout()
plt.show(block = True)

#####

plt.figure(figsize=(9,6))
sns.lineplot(
    data=senders_age_income,
    x="age_group",
    y="share_senders",
    hue="group",
    marker="o",
    linewidth=2.5
)

plt.title("Distribution of remittance senders by age and origin income group")
plt.ylabel("Share of remittances sent by each age group (%)")
plt.xlabel("Age group")
plt.grid(True, alpha=0.3)
plt.legend(title="Income group of origin", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show(block = True)
