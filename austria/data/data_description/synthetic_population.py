import pandas as pd

fixed_vars = ['agent_id', 'country', 'sex']

df_raw = pd.read_pickle("c:\\data\\population\\austria\\simulated_migrants_populations_2010-2024.pkl")
df_raw = df_raw[fixed_vars + [str(x) for x in range(2010, 2026)]]
df_raw.columns = fixed_vars + [str(x) for x in range(2010, 2026)]
df_raw = pd.melt(df_raw, id_vars=fixed_vars, value_vars=df_raw.columns[3:],
             value_name='age', var_name='year')
df_raw['year'] = df_raw['year'].astype(int)

df_filtered = df_raw[df_raw['year'].between(2013, 2023)].copy()

# Step 3: Identify agents present in each year
presence_counts = df_filtered.dropna(subset=['age']).groupby('year')['agent_id'].nunique()
print("Number of agents present each year from 2013 to 2023:")
print(presence_counts)

# Step 4: Determine agents present in 2013 and 2023
agents_2013 = set(df_filtered[df_filtered['year'] == 2013].dropna()['agent_id'])
agents_2023 = set(df_filtered[df_filtered['year'] == 2023].dropna()['agent_id'])

# Step 5: Calculate agents who left between 2013 and 2023
left_agents = agents_2013 - agents_2023
number_left = len(left_agents)
print(f"Number of agents who left between 2013 and 2023: {number_left}")

# Step 6: Identify new agents arrived between 2013 and 2023
agents_arrived = set(df_filtered.dropna()['agent_id']) - agents_2013
number_arrived = len(agents_arrived)
print(f"Number of new agents arrived between 2013 and 2023: {number_arrived}")

# Step 7: Count the total number of unique agents from 2013 to 2023
total_agents = len(set(df_filtered.dropna()['agent_id']))
print(f"Total number of unique agents from 2013 to 2023: {total_agents}")