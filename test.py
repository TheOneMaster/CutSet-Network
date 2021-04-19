import pandas as pd
import numpy as np
import CutSetNetwork as csn

# df = pd.read_csv("data\diabetes.csv")
# df['class'] = df['class'].map({"tested_positive": 1, "tested_negative": 0})
# network = csn.CutSetNetwork(df)

df2 = pd.read_csv(r"data\bank-additional-full.csv", sep=';')
df2['y'] = df2['y'].map({'no': 0, 'yes': 1})
network = csn.CutSetNetwork(df2)

print(network.tree)
