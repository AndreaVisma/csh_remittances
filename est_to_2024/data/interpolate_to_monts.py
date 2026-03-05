
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import re

df_all = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\synthetic_pop_mig_4obs.pkl")

df_all['date'] = pd.to_datetime(df_all['date'])
df_all.sort_values('date', inplace=True)

# Precompute mean_age lookup once (not per-row in a loop)
unique_age_groups = df_all['age_group'].unique()

list_df_months = []
pbar = tqdm(df_all.destination.unique())

for dest_country in pbar:
    pbar.set_description(f"Processing {dest_country}")
    df_dest = df_all[df_all.destination == dest_country]

    start_date = df_dest['date'].min()
    end_date = df_dest['date'].max()
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    monthly_times = (monthly_dates - start_date).days.to_numpy()

    # Compute time once per dest block
    times = (df_dest['date'] - start_date).dt.days.to_numpy()
    n_people = df_dest['n_people'].to_numpy()

    # Group by all three keys at once — no repeated boolean filtering
    group_keys = ['origin', 'age_group', 'sex']
    groups = df_dest.groupby(group_keys)

    records = []
    n_months = len(monthly_dates)

    for (origin, age_group, sex), grp_idx in groups.groups.items():
        grp = df_dest.loc[grp_idx]
        t = (grp['date'] - start_date).dt.days.to_numpy()
        y = grp['n_people'].to_numpy()

        try:
            cs = CubicSpline(t, y)
            vals = cs(monthly_times).astype(int)
        except Exception:
            continue  # skip silently instead of appending empty df

        records.append({
            'date': monthly_dates,
            'origin': origin,
            'age_group': age_group,
            'sex': sex,
            'n_people': vals,
            'destination': dest_country,
        })

    if not records:
        continue

    # Build one df per destination from columnar arrays
    df_month = pd.DataFrame({
        'date': np.tile(monthly_dates, len(records)),
        'origin': np.repeat([r['origin'] for r in records], n_months),
        'age_group': np.repeat([r['age_group'] for r in records], n_months),
        'sex': np.repeat([r['sex'] for r in records], n_months),
        'n_people': np.concatenate([r['n_people'] for r in records]),
        'destination': dest_country,
    })

    list_df_months.append(df_month)

df_months = pd.concat(list_df_months, ignore_index=True)
df_months['mean_age'] = df_months['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))

#####
# check us mex
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
mex = df_months[df_months.destination == 'USA'][['origin', 'n_people', 'date']].groupby(['origin', 'date']).sum().reset_index()
fig = px.line(mex, x = 'date', y = 'n_people', color='origin')
fig.show()

df_months.to_pickle("C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_interpolated_2010_to_2024.pkl")
