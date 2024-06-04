# Write your first Python code in Geoweaver
from joblib import dump, load
import os
homedir = os.path.expanduser('~')

# Load the model from the file
clf_dt = load(f'{homedir}/research/disease-prediction/modal.joblib')
# Load the test data from files
X_test = load(f'{homedir}/research/disease-prediction/X_test.joblib')
y_test = load(f'{homedir}/research/disease-prediction/y_test.joblib')




disease_pred = clf_dt.predict(X_test)
disease_real = y_test.values

for i in range(0, len(disease_real)):
    if disease_pred[i]!=disease_real[i]:
        print ('Pred: {0}\nActual: {1}\n'.format(disease_pred[i], disease_real[i]))
