"""
Script: weekly_nat_disasters.py
Author: Andrea Vismara
Date: 16/01/2025
Description: creates a global dataset of natural disaster events at the weekly level
"""

#imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import os
import json
pio.renderers.default = 'browser'

dict_names = {}
with open('c:\\data\\general\\countries_dict.txt',
          encoding='utf-8') as f:
    data = f.read()
js = json.loads(data)
for k,v in js.items():
    for x in v:
        dict_names[x] = k

#read the EM-DAT data
emdat = pd.read_excel("c:\\data\\natural_disasters\\emdat_2024_07_all.xlsx")
emdat = emdat[(emdat["Start Year"] >= 2010) &
              (emdat["Disaster Group"] == "Natural")].copy()
emdat["Country"] = emdat["Country"].map(dict_names)

#create a dataset with dummy variables
df_nd = pd.pivot_table(data=emdat, columns =["Disaster Type"],
                               index=["Country", "Start Year", "Start Month", "Start Day",
                                      "End Year", "End Month", "End Day"], values="Total Affected",
                           aggfunc="sum")
df_nd.fillna(0, inplace = True)
df_nd = df_nd.astype(int)
df_nd["total affected"] = df_nd.sum(axis=1)
df_nd.reset_index(inplace = True)

#clean dates
df_nd[['Start Year','Start Month', 'Start Day']] = (
    df_nd[['Start Year','Start Month', 'Start Day']].fillna(1).astype(int))
df_nd[['End Year','End Month', 'End Day']] = (
    df_nd[['End Year','End Month', 'End Day']].fillna(1).astype(int))
df_nd["start_date"] = pd.to_datetime(df_nd[['Start Year','Start Month', 'Start Day']].rename(columns=dict(zip(['Start Year','Start Month', 'Start Day'],
                                                                                                              ['year', 'month', 'day']))))
df_nd["end_date"] = pd.to_datetime(df_nd[['End Year','End Month', 'End Day']].rename(columns=dict(zip(['End Year','End Month', 'End Day'],
                                                                                                              ['year', 'month', 'day']))))
df_nd.drop(columns = ['End Year','End Month', 'End Day', 'Start Year','Start Month', 'Start Day'], inplace = True)
df_nd.rename(columns = {'Country' : 'country', "total affected" : "total_affected"}, inplace = True)

df_nd.to_excel("C:\\Data\\natural_disasters\\disasters_start_end_dates.xlsx", index = False)
