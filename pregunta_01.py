from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

#obtener data del dataset
wine = fetch_ucirepo(id=109)
X = wine.data.features
y = wine.data.targets

#variables predictoras
predictors = ["Alcohol", "Alcalinity_of_ash", "Nonflavanoid_phenols"]
X = X[predictors]

#particionar la base en 70% para train y 30% para test
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.3, random_state=42)

#modelo de random forest
rf_model = RandomForestClassifier(random_state=42)

#entrenar el modelo con los datos de entrenamiento
rf_model.fit(X_train, y_train)

#predicciones con los datos de test
y_pred = rf_model.predict(X_test)

#métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)