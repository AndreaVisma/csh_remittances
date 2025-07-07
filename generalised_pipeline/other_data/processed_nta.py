
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from utils import dict_names

nta_file = "C:\\Data\\economic\\nta\\NTA profiles.xlsx"

c_vals = pd.read_excel(nta_file, sheet_name='C')
yl_vals = pd.read_excel(nta_file, sheet_name='YL')


def work_df(df):
    cols = df.columns[11:].tolist()
    # cols = [re.findall(r'\d+', x) for x in cols]
    cols.append('country')
    df = df[cols]
    df = pd.melt(df, id_vars='country', value_name='value', var_name='age')
    df['age'] = df['age'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
    df['country'] = df.country.map(dict_names)
    return df

c_vals = work_df(c_vals).rename(columns = {'value' : 'consumption'})
yl_vals = work_df(yl_vals).rename(columns = {'value' : 'labour_income'})

df = c_vals.merge(yl_vals, on = ['country', 'age'])
df['nta'] = df['labour_income'] / df['consumption']

#####
import matplotlib.pyplot as plt
at = df[df.country == 'Austria']

fig, ax = plt.subplots(figsize = (5,5))
plt.plot(at.age, at.labour_income, linewidth = 3)
plt.plot(at.age, at.consumption, linewidth = 3)
plt.grid(True)
fig.savefig('.\plots\\for_paper\\AUSTRIA_NTA.png', bbox_inches = 'tight')
plt.show(block = True)

####

wb_groups = pd.read_excel("C://data//economic//income_classification_countries_wb.xlsx")[['country', 'group', 'Region']]
wb_groups['country'] = wb_groups.country.map(dict_names)
df = df.merge(wb_groups, on = 'country', how = 'left')
print(df.isna().sum())

###################
# extend NTA to countries for which we dont have it
###################

all_dest = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors.pkl").destination.unique().tolist()
dest_with_nta = df.country.unique().tolist()
missing_countries = list(set(all_dest) - set(dest_with_nta))

df_t = df.pivot_table(values='nta', columns='age', index=['country', 'group', 'Region']).reset_index()

list_dfs = []
for country in tqdm(missing_countries):
    group = wb_groups[wb_groups.country == country].group.item()
    region = wb_groups[wb_groups.country == country].Region.item()

    try:
        df_t_here = df_t[(df_t.group == group) & (df_t.Region == region)].iloc[:, 3:]
        if len(df_t_here) == 0:
            df_t_m = df_t[(df_t.group == group)].iloc[:, 3:]
            list_mean = [country, group, region] + df_t_m.mean().tolist()
        else:
            list_mean = [country, group, region] + df_t_here.mean().tolist()
        df_mean = pd.DataFrame([list_mean], columns=df_t.columns)
        list_dfs.append(df_mean)
        assert df_mean.isna().sum(1).item() < (len(df_mean.columns) -3)
    except:
        print(f"something wrong with {country}")

additional_dfs = pd.concat(list_dfs)
additional_dfs_t = additional_dfs.melt(id_vars=['country', 'group', 'Region'], value_vars=additional_dfs.columns[3:],
                           value_name='nta', var_name='age')

df = pd.concat([df, additional_dfs_t])
df.to_pickle("C:\\Data\\economic\\nta\\processed_nta.pkl")
###################
import matplotlib.pyplot as plt
df_nta = pd.read_pickle("C:\\Data\\economic\\nta\\processed_nta.pkl")
df_ita = df_nta[df_nta.country == "Saudi Arabia"]

plt.figure(figsize=(10, 6))
plt.plot(df_ita['age'], df_ita['nta'], linewidth=3)
plt.grid(True)
plt.show(block = True)




###################


