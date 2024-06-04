# Write your first Python code in Geoweaver
from sklearn import preprocessing
import pandas as pd
import os
import numpy as np

homedir = os.path.expanduser('~')
df = pd.read_csv(f'{homedir}/research/disease-prediction/cleaned_data.csv')
n_unique = len(df['symptom'].unique())
print(n_unique)
print(df.dtypes)

# Encode the Labels
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['symptom'])
print(integer_encoded)

# # One Hot Encode the Labels
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

# print
# print(onehot_encoded[0])
print('length of onehot encoded',len(onehot_encoded[0]))

cols = np.asarray(df['symptom'].unique())
print(cols)

# Create a new dataframe to save OHE labels
df_ohe = pd.DataFrame(columns = cols)
print(df_ohe.head())


for i in range(len(onehot_encoded)):
    df_ohe.loc[i] = onehot_encoded[i]

print(df_ohe.head())

print('ohe', len(df_ohe))

# Disease Dataframe
df_disease = df['disease']
print(df_disease.head())

# Concatenate OHE Labels with the Disease Column
df_concat = pd.concat([df_disease,df_ohe], axis=1)
print(df_concat.head())


df_concat.drop_duplicates(keep='first',inplace=True)
print(df_concat.head())
print(len(df_concat))

cols = df_concat.columns
print(cols)
cols = cols[1:]

# Since, every disease has multiple symptoms, combine all symptoms per disease per row
df_concat = df_concat.groupby('disease').sum()
df_concat = df_concat.reset_index()
print(df_concat[:5])

df_concat.to_csv(f'{homedir}/research/disease-prediction/training-data.csv', index=False)
print('training data successfully created')


