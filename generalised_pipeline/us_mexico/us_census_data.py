
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from utils import dict_names

df_deno = pd.read_csv("C:\\Data\\general\\demonyms.csv")
df_deno.columns = ["demonym", "country"]

us_folder = "C:\\Data\\migration\\bilateral_stocks\\us\\"

years = [2010, 2015, 2021]

df_all = pd.DataFrame([])
for year in tqdm(years):
    df_year = pd.read_excel(us_folder + f"us_{year}.xlsx", sheet_name="Data", skiprows=1)
    df_year = df_year[[x for x in df_year.columns if "Unnamed" not in x]].iloc[1:]
    # cols_to_keep = ['sex', 'age_group']
    # other_cols = [x for x in df_year.columns if "any" in x]
    # cols_to_keep = cols_to_keep + other_cols
    # df_year = df_year[cols_to_keep]
    df_year = df_year.melt(id_vars=["sex", "age_group"], value_name='n_people', var_name='denomination')

    df_year['demonym'] = df_year['denomination'].apply(lambda x: x.split(" ")[0])
    ## special cases
    df_year.loc[df_year.denomination.str.contains("Asian Indian"), 'demonym'] = "Indian"
    df_year.loc[df_year.denomination.str.contains("Sri Lanka"), 'demonym'] = "Sri Lankan"
    df_year.loc[df_year.denomination.str.contains("Puerto Rican"), 'demonym'] = "Puerto Rican"
    df_year.loc[df_year.denomination.str.contains("Costa Rican"), 'demonym'] = "Costa Rican"

    df_year = df_year.merge(df_deno, on = 'demonym', how = 'left')
    df_year = df_year[~df_year.country.isna()]
    df_year['country'] = df_year.country.map(dict_names)
    df_year = df_year[~df_year.country.isna()]
    df_year = df_year[['country', 'sex', 'age_group', 'n_people']]
    df_year['year'] = year

    df_all = pd.concat([df_all, df_year])

df_all['n_people'] = df_all.n_people.str.replace(',', '')
df_all['n_people'] = df_all.n_people.astype(int)
df_all = df_all[df_all.country != 'USA']

df_all = df_all[df_all.age_group != 'total']
df_all['age_group'] = df_all.age_group.str.replace('and ove', '100')
df_all['mean_age'] = df_all['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))

df_all.rename(columns = {"country" : "origin"}, inplace = True)
df_all['destination'] = "USA"
df_all = df_all[["origin", "destination", "sex", "age_group", "n_people", "mean_age", "year"]]

df_all.to_excel("C:\\Data\\migration\\bilateral_stocks\\us\\processed_asia_latam.xlsx", index = False)

def plot_pyramid_country_year(country, year):

    df_ = df_all[(df_all.origin == country) & (df_all.year == year)]
    df_ = df_[['sex', 'mean_age', 'n_people']].pivot_table(index='mean_age', columns='sex', values='n_people').reset_index()

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(df_["mean_age"], -df_["male"], color="blue", label="Male")
    ax.barh(df_["mean_age"], df_["female"], color="red", label="Female")

    ax.set_xlabel("Population Count")
    ax.set_ylabel("Age Group")
    ax.set_title(f"Population pyramid of the diaspora from {country} living in the US in {year}")

    # xticks = np.linspace(-max(df["male"].max(), df["female"].max()),
    #                      max(df["male"].max(), df["female"].max()),
    #                      num=5)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels([str(abs(int(x))) for x in xticks])

    ax.legend()
    plt.grid()
    plt.show(block = True)

plot_pyramid_country_year("Japan", 2010)