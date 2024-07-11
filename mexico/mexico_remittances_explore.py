"""
Script: mexico_remittances_explore.py
Author: Andrea Vismara
Date: 10/07/2024
Description: Explores the data for the remittances inflow in mexico
"""

##imports
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

plot_series = True #change this to plot time series or not
out_folder = "c:\\git-projects\\csh_remittances\\mexico\\plots\\summary_remittances\\"

##load the data in
df = pd.read_excel("c:\\data\\remittances\\mexico\\mexico_remittances_2024.xlsx",
                   skiprows=9)
df = df.iloc[8:,:]
names_column = ["date",
                "total_mln", "money_orders_mln", "checks_mln", "electronic_transfers_mln", "cash_goods_mln",
                "total_operations", "money_orders_operations", "checks_operations", "electronic_transfers_operations", "cash_goods_operations",
                "total_mean_op", "money_orders_mean_op", "checks_mean_op", "electronic_transfers_mean_op", "cash_goods_mean_op"]
dict_columns = dict(zip(df.columns,names_column))
df.rename(columns = dict_columns, inplace = True)
df.to_excel("c:\\data\\remittances\\mexico\\remittances_renamed.xlsx", index = False)

##plot time series for every column
if plot_series:
    #########
    # totals
    #########
    fig = go.Figure()
    past_values = 0
    for col in df.columns[2:6]:
        y = df[col] + past_values
        fig.add_trace(go.Scatter(x=df.date, y=y, fill='tonexty',
                                 mode = "markers+lines", name = col))  # fill down to xaxis
        past_values = y
    fig.add_trace(go.Scatter(x=df.date, y=df.total_mln,
                                 mode = "lines", name = "total", line= dict(color = 'pink')))
    fig.update_layout(title = "Total remittances sent by type of transfer")
    fig.update_yaxes(title = "Mln of US Dollars")
    fig.write_html(out_folder + "total_remittances_by_type_line.html")
    fig.show()

    #shares in total
    df_tot_shares = df[["date"] + df.columns[1:6].tolist()].copy()
    for col in df.columns[2:6]:
        df_tot_shares[col] = 100 * df_tot_shares[col] / df_tot_shares["total_mln"]
    fig = go.Figure()
    for col in df.columns[2:6]:
        fig.add_trace(go.Bar(x=df_tot_shares.date, y=df_tot_shares[col], name = col))
    fig.update_layout(title = "Total remittances sent by type of transfer, shares",
                      barmode = 'stack')
    fig.update_yaxes(title = "Share in the total of remittances sent", ticksuffix="%")
    fig.write_html(out_folder + "total_remittances_share_by_type.html")
    fig.show()

    #########
    # number of operations
    #########
    fig = go.Figure()
    past_values = 0
    for col in df.columns[7:11]:
        y = df[col] + past_values
        fig.add_trace(go.Scatter(x=df.date, y=y, fill='tonexty',
                                 mode = "markers+lines", name = col))  # fill down to xaxis
        past_values = y
    fig.update_layout(title = "Number of operations by type of transfer")
    fig.update_yaxes(title = "Thousands of operations")
    fig.write_html(out_folder + "number_of_operations_by_type_line.html")
    fig.show()

    #shares
    df_tot_shares = df[["date"] + df.columns[7:11].tolist()].copy()
    for col in df.columns[7:11]:
        df_tot_shares[col] = 100 * df_tot_shares[col] / df["total_operations"]
    fig = go.Figure()
    for col in df.columns[7:11]:
        fig.add_trace(go.Bar(x=df_tot_shares.date, y=df_tot_shares[col], name = col))
    fig.update_layout(title = "Total operations by type of transfer, shares",
                      barmode = 'stack')
    fig.update_yaxes(title = "Share in the total number of operations", ticksuffix="%")
    fig.write_html(out_folder + "total_operations_share_by_type.html")
    fig.show()

    #########
    # mean value of operations
    #########
    fig = go.Figure()
    past_values = 0
    for col in df.columns[12:]:
        fig.add_trace(go.Scatter(x=df.date, y=df[col],
                                 mode = "markers+lines", name = col))  # fill down to xaxis
    fig.update_layout(title = "Mean amount per operation, by type")
    fig.update_yaxes(title = "US Dollars")
    fig.write_html(out_folder + "mean_amount_per_operation_by_type_line.html")
    fig.show()

##########################
# load the disaster data
##########################

emdat = pd.read_excel("c:\\data\\natural_disasters\\emdat_2024_07_all.xlsx")
emdat = emdat[(emdat.Country == "Mexico") &
              (~emdat["Total Affected"].isna()) &
              (emdat["Disaster Group"] == "Natural") &
              (emdat["Start Year"] >= 1995)].copy()
emdat.sort_values("Total Affected", ascending = False, inplace = True)

emdat['Disaster Type'].value_counts()

## data for high intensity events
emdat_high_aff = emdat[emdat["Total Affected"] >= 50_000]
emdat_high_aff = emdat_high_aff[['Disaster Type', 'Start Year',
       'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day', 'Total Affected']]

emdat_high_aff[['Start Year','Start Month', 'Start Day']] = (
    emdat_high_aff[['Start Year','Start Month', 'Start Day']].fillna(1).astype(int))
emdat_high_aff.rename(columns = dict(zip(['Start Year','Start Month', 'Start Day'],
                              ["year", "month", "day"])), inplace = True)
emdat_high_aff["date_start"] = pd.to_datetime(emdat_high_aff[["year", "month", "day"]])
emdat_high_aff.drop(columns = ["year", "month", "day"], inplace = True)

emdat_high_aff[['End Year','End Month', 'End Day']] = (
    emdat_high_aff[['End Year','End Month', 'End Day']].fillna(12).astype(int))
emdat_high_aff.rename(columns = dict(zip(['End Year','End Month', 'End Day'],
                              ["year", "month", "day"])), inplace = True)
emdat_high_aff["date_end"] = pd.to_datetime(emdat_high_aff[["year", "month", "day"]])
emdat_high_aff.drop(columns = ["year", "month", "day"], inplace = True)

## plot remittances and disasters occurrence

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(
    x = df.date, y = df.total_mln, name = "Total remittances"
), secondary_y=False)

colors = ['red', 'yellow', 'green', 'brown', 'pink']
colors_dict = dict(zip(emdat_high_aff["Disaster Type"].unique(), colors))
for disaster in emdat_high_aff["Disaster Type"].unique():
    df_dis = emdat_high_aff[emdat_high_aff["Disaster Type"] == disaster]
    for i in range(len(df_dis)):
        fig.add_vline(x=df_dis.iloc[i]["date_start"],
                      name = disaster,
                      line_width=2, line_color=colors_dict[disaster],
                      showlegend = True if i == 0 else False)
fig.show()


