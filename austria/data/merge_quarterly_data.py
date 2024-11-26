import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import austria_nighbours, dict_names

#population
df_pop = pd.read_excel("c:\\data\\migration\\austria\\quarterly_population_clean.xlsx")

#remittances
df_rem = pd.read_excel("c:\\data\\remittances\\austria\\quarterly_remittances_sent_clean.xlsx")
df = df_pop.merge(df_rem, on = ['country', 'year', 'quarter'], how = 'inner')

#dependency_ratio
df_age = pd.read_excel("c:\\data\\population\\austria\\age_nationality_hist_quarterly.xlsx")
df = df.merge(df_age, on = ['country', 'year', 'quarter'], how = 'inner')

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
for country in tqdm(df.country.unique()):
    df.loc[df.country == country, 'students'] = df.loc[df_rem.country == country, 'students'].ffill()
df['students'].fillna(0, inplace = True)
df['pct_students'] = 100 * df['students'] / df['population']
df['pct_students'] = df['pct_students'].clip(0,100)

df.to_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx", index = False)