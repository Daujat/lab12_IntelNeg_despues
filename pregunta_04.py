import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('glioma_grading-1.csv', sep=';')

#características X y variable objetivo y
X = data.drop(['Grade'], axis=1)
y = data['Grade']

#codificación one-hot en las variables categoricas
categorical_features = ['Race']
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(X[categorical_features]).toarray()
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
X_encoded = pd.concat([X.drop(categorical_features, axis=1), pd.DataFrame(encoded_features, columns=encoded_feature_names)], axis=1)

#dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#modelo de random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#predicciones con random forest en el conjunto de prueba
rf_pred = rf_model.predict(X_test)

#metricas de evaluación para random forest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')

print("Random Forest:")
print("Accuracy:", rf_accuracy)
print("F1 Score:", rf_f1)

#modelo de SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

#predicciones con SVM en el conjunto de prueba
svm_pred = svm_model.predict(X_test)

#métricas de evaluación para SVM
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='weighted')

print("\nSVM:")
print("Accuracy:", svm_accuracy)
print("F1 Score:", svm_f1)