import pandas as pd

data = pd.read_csv('aids_clinical.csv', sep=';')

#correlación entre age y wtkg
correlation = data['age'].corr(data['wtkg'])

print("Coeficiente de correlación entre 'age' y 'wtkg':", correlation)