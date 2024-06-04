# Write your first Python code in Geoweaver
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('~/research/disease-prediction/raw_data.xlsx')
print(df.head())
data = df.fillna(method='ffill')
print('after filling null values\n')
print(data.head())
data.to_csv('~/research/disease-prediction/data-withoutnull')

