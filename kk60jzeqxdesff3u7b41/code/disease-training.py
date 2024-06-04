# Write your first Python code in Geoweaver
# Write your first Python code in Geoweaver
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import os
from joblib import dump, load

homedir = os.path.expanduser('~')

df = pd.read_csv(f'{homedir}/research/disease-prediction/training-data.csv')
cols = df.columns[1:]
print("symptoms")
print(cols)

# One Hot Encoded Features
X = df[cols]

# Labels
y = df['disease']


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

print(len(X_train), len(y_train))

print(len(X_test), len(y_test))
# Save the test data to files
dump(X_test, f'{homedir}/research/disease-prediction/X_test.joblib')
dump(y_test, f'{homedir}/research/disease-prediction/y_test.joblib')

dt = DecisionTreeClassifier()
clf_dt=dt.fit(X_train, y_train)

clf_dt.score(X_train, y_train)

dump(clf_dt, f'{homedir}/research/disease-prediction/modal.joblib')

# export_graphviz(dt, 
#                 out_file='./tree.dot', 
#                 feature_names=cols)

# from graphviz import Source
from sklearn import tree

# graph image related
# graph = Source(export_graphviz(dt, 
#                 out_file=None, 
#                 feature_names=cols))

# png_bytes = graph.pipe(format='png')

# with open('tree.png','wb') as f:
#     f.write(png_bytes)

# from IPython.display import Image
# Image(png_bytes)

disease_pred = clf_dt.predict(X_test)
disease_real = y_test.values

for i in range(0, len(disease_real)):
    if disease_pred[i]!=disease_real[i]:
        print ('Pred: {0}\nActual: {1}\n'.format(disease_pred[i], disease_real[i]))

