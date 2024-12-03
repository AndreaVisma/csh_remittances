import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import austria_nighbours, dict_names, euro_area

#population
df_pop = pd.read_excel("c:\\data\\migration\\austria\\quarterly_population_clean.xlsx")

#remittances
df_rem = pd.read_excel("c:\\data\\remittances\\austria\\quarterly_remittances_sent_clean.xlsx")
df = df_pop.merge(df_rem, on = ['country', 'year', 'quarter'], how = 'inner')

#dependency_ratio
df_age = pd.read_excel("c:\\data\\population\\austria\\age_nationality_hist_quarterly.xlsx")
df = df.merge(df_age, on = ['country', 'year', 'quarter'], how = 'left')
for country in tqdm(df.country.unique()):
    df.loc[df.country == country, 'dep_ratio'] = df.loc[df.country == country, 'dep_ratio'].interpolate()

# dummy for neighbouring countries
df["neighbour_dummy"] = np.where(df["country"].isin(austria_nighbours), 1, 0)

## income category
df_class = pd.read_excel("c:\\data\\economic\\income_classification_countries_wb.xlsx", usecols="A:B", skipfooter=49)
df_class['country'] = df_class['country'].map(dict_names)
df = df.merge(df_class, on = 'country', how = 'left')

##GDP
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\quarterly_gdp_clean.xlsx")
for year in tqdm(df_gdp.year.unique(),
                          total = len(df_gdp.year.unique())):
    df_year = df_gdp[df_gdp.year == year]
    for quarter in df_year.quarter.unique():
        df_gdp.loc[(df_gdp.year == year) & (df_gdp.quarter == quarter), 'delta_gdp'] = (
                df_gdp.loc[(df_gdp.year == year) & (df_gdp.quarter == quarter), 'gdp_per_capita'] -
                df_gdp.loc[(df_gdp.year == year) & (df_gdp.quarter == quarter) & (df_gdp.country == 'Austria'), 'gdp_per_capita'].item())
df = df.merge(df_gdp, on = ['country', 'year', 'quarter'], how = 'left')
for country in tqdm(df.country.unique()):
    df.loc[df.country == country, 'gdp_per_capita'] = (
        df.loc[df.country == country, 'gdp_per_capita'].interpolate())
    df.loc[df.country == country, 'delta_gdp'] = (
        df.loc[df.country == country, 'delta_gdp'].interpolate())
df.dropna(inplace = True)

#natural disasters
df_nd = pd.read_excel("C:\\Data\\natural_disasters\\emdat_country_type_quarterly.xlsx")
#clean dates
df_nd[['Start Year','Start Month', 'Start Day']] = (
    df_nd[['Start Year','Start Month', 'Start Day']].fillna(1).astype(int))
df_nd.rename(columns = dict(zip(['Start Year','Start Month', 'Start Day'],
                              ["year", "month", "day"])), inplace = True)
df_nd["date_start"] = pd.to_datetime(df_nd[["year", "month", "day"]])
df_nd.drop(columns = ["year", "month", "day", "quarter"], inplace = True)
df_nd['year'] = df_nd['date_start'].dt.year
df_nd['quarter'] = df_nd['date_start'].dt.quarter
df_nd['quarter_after_1_month'] = df_nd['date_start'].apply(lambda x: 1 + (x.month)//3)
df_nd.rename(columns = {'Country' : 'country'}, inplace = True)
## country population
df_pop_country = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")
#merge
df_nd = df_nd.merge(df_pop_country, on=['country', 'year'], how = 'left')

#percentage affected dataframe
df_nd_pct = df_nd.copy()
cols = ['Animal incident', 'Drought', 'Earthquake', 'Epidemic',
       'Extreme temperature', 'Flood', 'Glacial lake outburst flood', 'Impact',
       'Infestation', 'Mass movement (dry)', 'Mass movement (wet)', 'Storm',
       'Volcanic activity', 'Wildfire', 'total affected']
for col in cols:
    df_nd_pct[col] = 100 * df_nd_pct[col] / df_nd_pct['population']
df_nd_pct.dropna(inplace = True)
cols = ['country', 'Animal incident', 'Drought', 'Earthquake', 'Epidemic',
       'Extreme temperature', 'Flood', 'Glacial lake outburst flood', 'Impact',
       'Infestation', 'Mass movement (dry)', 'Mass movement (wet)', 'Storm',
       'Volcanic activity', 'Wildfire', 'total affected', 'year',
       'quarter']
df_group = df_nd_pct[cols].groupby(['country', 'year', 'quarter']).sum().reset_index()
df = df.merge(df_group, left_on = ['country', 'year', 'quarter'],
                  right_on = ['country', 'year', 'quarter'], how = 'left')
df.fillna(0, inplace = True)

##growth rate of remittances
df = df.sort_values(by=['country', 'year', 'quarter'])
df['growth_rate_rem'] = df.groupby('country')['remittances'].pct_change() * 100  # Multiply by 100 for percentage format
df.replace([np.inf, -np.inf], 0, inplace=True)

df.dropna(inplace =True)

##students
df_stud = pd.read_excel("c:\\data\\population\\austria\\students_by_origin_clean.xlsx")
df = df.merge(df_stud, on = ['country', 'year', 'quarter'], how = 'left')
df = df[~((df.year == 2010) & (df.quarter == 2))]
for country in tqdm(df.country.unique()):
    df.loc[df.country == country, 'students'] = df.loc[df.country == country, 'students'].interpolate()
df['students'].fillna(0, inplace = True)
df['pct_students'] = 100 * df['students'] / df['population']
df['pct_students'] = df['pct_students'].clip(0,100)

##cost data
df_cost = pd.read_excel("C:\\Data\\remittances\\remittances_cost_from_euro.xlsx")
df_cost.rename(columns = {"destination_name" : "country", "period" : "year"}, inplace = True)
df_cost = df_cost[['year', 'country', 'pct_cost']].groupby(['year', 'country']).mean().reset_index()

df = df.merge(df_cost, on= ["country", "year"], how = "left")
# give to all countries in a certain income group the same cost
for group in tqdm(df_class['group'].unique()):
    group_countries = df_class[df_class.group == group]["country"].unique().tolist()
    for year in df.year.unique():
        mean_year_group = df_cost.loc[(df_cost.country.isin(group_countries)) & (df_cost.year == year),
        "pct_cost"].mean()
        df.loc[(df.country.isin(group_countries)) & (df.year == year) & (df.pct_cost.isna()),
        "pct_cost"] = mean_year_group
#euro area countries have no cost
df.loc[df.country.isin(euro_area), "pct_cost"] = 0

##inflation
df_inf = pd.read_excel("C:\\Data\\economic\\annual_inflation_clean.xlsx")
df_inf.rename(columns = {"Country" : "country"}, inplace = True)
df_inf.loc[df_inf['hcpi'] > 500, 'hcpi'] = 500
df_inf['quarter'] = 1
df = df.merge(df_inf, on= ["country", "year", "quarter"], how = "left")
df = df[df.year > 2010]
## interpolate inflation
for country in tqdm(df.country.unique()):
    df.loc[df.country == country, "hcpi"] = df.loc[df.country == country, "hcpi"].interpolate()

## gdp growth rate
df['growth_rate_gdp'] = df.groupby('country')['gdp_per_capita'].pct_change() * 100

df.dropna(inplace = True)
df.to_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx", index = False)