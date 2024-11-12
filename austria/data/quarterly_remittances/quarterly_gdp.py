import pandas as pd
from utils import dict_names
from tqdm import tqdm
from utils import get_quarter, get_last_day_of_the_quarter

df = pd.read_excel("C:\\Data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx")
df.year = pd.to_datetime(df.year, format="%Y").map(get_last_day_of_the_quarter)

# Prepare DataFrame for quarterly interpolation
df_q = pd.DataFrame()
for country in tqdm(df['country'].unique()):
    df_country = df[df['country'] == country].copy()
    df_country.set_index(["year"], inplace=True)
    df_country = df_country.asfreq('Q')
    df_country['gdp_per_capita'] = df_country['gdp'].resample("m").interpolate(method="time")
    df_country['country'] = country
    df_q = pd.concat([df_q, df_country])

df_q = df_q.reset_index()
df_q['quarter'] = df_q.year.map(get_quarter)
df_q['year'] = df_q.year.apply(lambda x: x.year)
df_q.drop(columns='gdp', inplace = True)

df_q.to_excel("c:\\data\\economic\\gdp\\quarterly_gdp_clean.xlsx", index = False)