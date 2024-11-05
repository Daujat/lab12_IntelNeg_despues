import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('breast_wisconsin.csv', sep=';')

#separar las características X y la variable objetivo y
X = data.drop(['COD_identificacion_dni', 'fractal_dimension3'], axis=1)
y = data['fractal_dimension3']

#dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#modelo de random forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#predicciones en el conjunto de prueba
y_pred = rf_model.predict(X_test)

#metricas de evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Error Cuadrático Medio (MSE):", mse)
print("Coeficiente de Determinación (R^2):", r2)