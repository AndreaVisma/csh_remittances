
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from utils import dict_names

ger_new = "C:\\Users\\Andrea Vismara\\Downloads\\ger_new\\12521-0003_en_flat.csv"

df = pd.read_csv(ger_new, sep = ';', encoding="UTF-8-SIG")
df.rename(columns = dict(zip(["2_variable_attribute_label", "3_variable_attribute_label", "4_variable_attribute_label", "value"],
                             ["age_group", "origin", "sex", "n_people"])), inplace = True)
df = df[["time", "origin", "age_group", "sex", "n_people"]]
df = df[df.sex != "Total"]
df['origin'] = df['origin'].map(dict_names)
df.dropna(inplace = True)

df['age'] = df['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
df["age_group"] = pd.cut(df.age, bins = bins).map(dict(zip(bins, labels)))

df.to_pickle("C:\\Data\\migration\\bilateral_stocks\\germany\\processed_germany.pkl")

df_ = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\germany\\processed_germany.pkl")




