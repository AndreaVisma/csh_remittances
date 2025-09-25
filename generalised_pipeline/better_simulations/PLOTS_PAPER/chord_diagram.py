
import pandas as pd
from pycirclize import Circos
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
dict_names = {}
with open('c:\\data\\general\\countries_dict.txt',
          encoding='utf-8') as f:
    data = f.read()
js = json.loads(data)
for k,v in js.items():
    for x in v:
        dict_names[x] = k

# Example dataframe (flow data)
df_with = pd.read_parquet("./general/results_plots/all_flows_simulations_with_disasters_CORRECT.parquet")
df_without = pd.read_parquet("./general/results_plots/all_flows_simulations_without_disasters_CORRECT.parquet")

df = (df_with[["origin", "destination", "sim_remittances"]].groupby(["origin", "destination"]).
      sum().sort_values("sim_remittances").reset_index())
df["sim_remittances"] /= 1e6
trillion_rem_total = df["sim_remittances"].sum() / 1e6

print(f"Total remittances moved between 2010 and 2019: {round(trillion_rem_total, 3)} trillion")

df_ = (df_without[["origin", "destination", "sim_remittances"]].groupby(["origin", "destination"]).
      sum().sort_values("sim_remittances").reset_index())
df_["sim_remittances"] /= 1e6

income_class = pd.read_excel("C:\\Data\\economic\\income_classification_countries_wb.xlsx")
income_class['country'] = income_class.country.map(dict_names)
income_class = income_class[["country", "group", "Region"]]

df = df.merge(income_class[["country", "group"]], left_on = "origin", right_on = "country")
df = df.merge(income_class[["country", "group"]], left_on = "destination", right_on = "country", suffixes = ("_origin", "_destination"))
df = df.merge(df_, on = ["origin", "destination"], suffixes = ("_with", "_without"))
df["disaster_rem"] = df["sim_remittances_with"] - df["sim_remittances_without"]
df.isna().sum()

df_groups = df[["group_origin", "group_destination", "sim_remittances_with", "sim_remittances_without", "disaster_rem"]].groupby(["group_origin", "group_destination"]).sum().reset_index()

#####################
# Sum remittances by origin group
receive_sums = (df_groups.groupby("group_origin")["sim_remittances_with"].sum().reset_index(name="total_sent"))
grand_total = receive_sums["total_sent"].sum()
receive_sums["pct_of_total"] = 100 * receive_sums["total_sent"] / grand_total
print(receive_sums)

sent_sums = (df_groups.groupby("group_destination")["sim_remittances_with"].sum().reset_index(name="total_sent"))
grand_total_sent = sent_sums["total_sent"].sum()
sent_sums["pct_of_total"] = 100 * sent_sums["total_sent"] / grand_total
print(sent_sums)

#####
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_clean.xlsx")
df_gdp['country'] = df_gdp.country.map(dict_names)
df_gdp['gdp'] /= 1e6
df_gdp = df_gdp.merge(income_class[["country", "group"]], on = "country")
df_gdp_group = df_gdp[(df_gdp["year"] > 2009) & (df_gdp.year < 2020)][["group", "gdp"]].groupby("group").mean().reset_index()

#
df_rec_pct = receive_sums[["group_origin", "total_sent"]].copy()
for group in df_gdp_group.group.unique():
    df_rec_pct.loc[df_rec_pct.group_origin == group, "total_sent"] /= df_gdp_group[df_gdp_group.group == group].gdp.item()

df_sent_pct = sent_sums[["group_destination", "total_sent"]].copy()
for group in df_gdp_group.group.unique():
    df_sent_pct.loc[df_sent_pct.group_destination == group, "total_sent"] /= df_gdp_group[
        df_gdp_group.group == group].gdp.item()

######################
# Plot chord diagram for groups
######################
all_groups = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
matrix_df = df_groups.pivot(index="group_destination", columns="group_origin", values= "sim_remittances_with")
matrix_df.to_excel('.\plots\\for_paper\\chord_income_groups.xlsx')

# Initialize Circos instance for chord diagram plot
circos = Circos.chord_diagram(
    matrix_df,
    space=3,
    cmap="plasma",
    label_kws=dict(size=12),
    link_kws=dict(direction=1, ec="black", lw=1),
)
fig = circos.plotfig()
fig.savefig('.\plots\\for_paper\\chord_income_groups.svg', bbox_inches = 'tight')
plt.show(block = True)

#################
matrix_dis = df_groups.pivot(index="group_destination", columns="group_origin", values= "disaster_rem")
matrix_dis.to_csv('.\plots\\for_paper\\chord_income_groups_disasters.csv')

# Initialize Circos instance for chord diagram plot
circos = Circos.chord_diagram(
    matrix_dis,
    space=3,
    cmap="plasma",
    label_kws=dict(size=12),
    link_kws=dict(direction=1, ec="black", lw=1),
)
fig = circos.plotfig()
fig.savefig('.\plots\\for_paper\\chord_income_groups_DISASTERS.svg', bbox_inches = 'tight')
plt.show(block = True)


############################
# Convert flow data to matrix format for chord_diagram
# Create a matrix where rows are origins and columns are destinations
all_countries = sorted(list(set(df["origin"]).union(set(df["destination"]))))

# Initialize matrix with zeros
matrix_data = []
for destination in tqdm(all_countries):
    row = []
    for origin in all_countries:
        # Find the flow amount for this origin-destination pair
        flow = df.loc[(df["destination"] == destination) & (df["origin"] == origin), "sim_remittances"]
        row.append(flow.sum() if not flow.empty else 0)
    matrix_data.append(row)

# Create matrix dataframe
matrix_df = pd.DataFrame(matrix_data, index=all_countries, columns=all_countries)

# Initialize Circos instance for chord diagram plot
circos = Circos.chord_diagram(
    matrix_df,
    cmap="viridis",
    # label_kws=dict(size=12),
    link_kws=dict(direction=1, ec="black", lw=0.5), ##inverse direction
)

print("Flow matrix:")
print(matrix_df)

# Plot
fig = circos.plotfig()
plt.show(block = True)
