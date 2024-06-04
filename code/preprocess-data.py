# Write your first Python code in Geoweaver
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import csv

data = pd.read_csv('~/research/disease-prediction/data-withoutnull')


def process_data(data):
    data_list = []
    data_name = data.replace('^','_').split('_')
    n = 1
    for names in data_name:
        if (n % 2 == 0):
            data_list.append(names)
        n += 1
    return data_list

disease_list = []
disease_symptom_dict = defaultdict(list)
disease_symptom_count = {}
count = 0

# if os.path.exists("myfile.dat"):
#     print("skip already exist")
# else:
#     f = file("myfile.dat", "w")

for idx, row in data.iterrows():
    
    # Get the Disease Names
    if (row['Disease'] !="\xc2\xa0") and (row['Disease'] != ""):
        disease = row['Disease']
        disease_list = process_data(data=disease)
        count = row['Count of Disease Occurrence']

    # Get the Symptoms Corresponding to Diseases
    if (row['Symptom'] !="\xc2\xa0") and (row['Symptom'] != ""):
        symptom = row['Symptom']
        symptom_list = process_data(data=symptom)
        for d in disease_list:
            for s in symptom_list:
                disease_symptom_dict[d].append(s)
            disease_symptom_count[d] = count
    
# file_path = os.path.expanduser('~/research/disease-prediction/cleaned_data.csv')
homedir = os.path.expanduser('~')
f = open(f'{homedir}/research/disease-prediction/cleaned_data.csv', 'w')

with f:
    writer = csv.writer(f)
    for key, val in disease_symptom_dict.items():
        for i in range(len(val)):
            writer.writerow([key, val[i], disease_symptom_count[key]])

df = pd.read_csv('~/research/disease-prediction/cleaned_data.csv')
df.columns = ['disease', 'symptom', 'occurence_count']
print(df.head())

# Remove any rows with empty values
df.replace(float('nan'), np.nan, inplace=True)
df.dropna(inplace=True)

df.to_csv(f'{homedir}/research/disease-prediction/cleaned_data.csv')


