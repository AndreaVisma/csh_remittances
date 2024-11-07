import pandas as pd
from utils import dict_names
from tqdm import tqdm
from utils import get_quarter, get_last_day_of_the_quarter

df = pd.read_excel("c:\\data\\migration\\austria\\population_by_nationality_year_land_2010-2024.xlsx")
df = df.melt(id_vars = 'year', value_vars=df.columns[2:],
             value_name='population', var_name='nationality')
df.loc[df['population'] == '-', 'population'] = 0
df = df.groupby(['year', 'nationality']).sum()
df = df.reset_index()
df['nationality'] = df['nationality'].map(dict_names)
df = df.dropna()
df.rename(columns = {'nationality':'country'}, inplace = True)
df.population = df.population.astype('float')
df.year = pd.to_datetime(df.year, format="%Y").map(get_last_day_of_the_quarter)

# Prepare DataFrame for quarterly interpolation
df_q = pd.DataFrame()
for country in tqdm(df['country'].unique()):
    print(f'{country}')
    df_country = df[df['country'] == country].copy()
    df_country.set_index(["year"], inplace=True)
    df_country = df_country.asfreq('Q')
    df_country['population'] = df_country['population'].resample("m").interpolate(method="time")
    df_country['country'] = country
    df_q = pd.concat([df_q, df_country])

df_q = df_q.reset_index()
df_q['quarter'] = df_q.year.map(get_quarter)
df_q['year'] = df_q.year.apply(lambda x: x.year)

df.to_excel("c:\\data\\migration\\austria\\quarterly_population_clean.xlsx")