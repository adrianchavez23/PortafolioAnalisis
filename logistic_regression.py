import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns

df = pd.read_csv('heart-disease.csv')
df.columns = df.columns.str.strip()

# Definir features y label
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
label = ['target']

df_X = df[features]
df_y = df[label]

# Convertir booleanos a enteros si es necesario
for col in df_X.columns:
    if df_X[col].dtype == 'bool':
        df_X[col] = df_X[col].astype(int)

# Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.25, random_state=42)

# Escalar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo 
regression = LogisticRegression()
regression.fit(X_train, y_train.values.ravel())

# Predicciones
y_pred = regression.predict(X_test)

# Evaluación en el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred)

# Evaluación en el conjunto de entrenamiento para diagnosticar bias
y_train_pred = regression.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Comparación de precisión (bias y varianza)
print(f'Precisión en el conjunto de entrenamiento: {train_accuracy:.2f}')
print(f'Precisión en el conjunto de prueba: {accuracy:.2f}')

# Diagnóstico de bias y varianza
if train_accuracy < 0.8:
    print("Bias: Alto")
elif 0.8 <= train_accuracy < 0.9:
    print("Bias: Medio")
else:
    print("Bias: Bajo")

if abs(train_accuracy - accuracy) > 0.1:
    print("Varianza: Alta")
else:
    print("Varianza: Baja")

# Matriz de confusión (antes de regularización)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=regression.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión (Sin Regularización)')
plt.show()

# Reporte de clasificación (antes de regularización)
print("Reporte de Clasificación (Sin Regularización):")
print(classification_report(y_test, y_pred))

# Gráfica de la distribución de las probabilidades predichas
probs = regression.predict_proba(X_test)[:, 1]
sns.histplot(probs, kde=True, bins=10)
plt.title('Distribución de las Probabilidades Predichas (Sin Regularización)')
plt.xlabel('Probabilidad Predicha')
plt.ylabel('Frecuencia')
plt.show()

# Regularización L2
regression_l2 = LogisticRegression(penalty='l2', C=0.1, solver='liblinear')  # C ajusta la fuerza de regularización
regression_l2.fit(X_train, y_train.values.ravel())

# Predicciones con regularización
y_pred_l2 = regression_l2.predict(X_test)

# Evaluación del modelo regularizado
l2_accuracy = accuracy_score(y_test, y_pred_l2)
print(f'Precisión del modelo con regularización L2: {l2_accuracy:.2f}')

# Matriz de confusión (con regularización)
cm_l2 = confusion_matrix(y_test, y_pred_l2)
disp_l2 = ConfusionMatrixDisplay(confusion_matrix=cm_l2, display_labels=regression.classes_)
disp_l2.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión (Con Regularización L2)')
plt.show()

# Reporte de clasificación (con regularización)
print("Reporte de Clasificación (Con Regularización L2):")
print(classification_report(y_test, y_pred_l2))

# Comparar la distribución de las probabilidades predichas (con regularización)
probs_l2 = regression_l2.predict_proba(X_test)[:, 1]
sns.histplot(probs_l2, kde=True, bins=10)
plt.title('Distribución de las Probabilidades Predichas (Con Regularización)')
plt.xlabel('Probabilidad Predicha')
plt.ylabel('Frecuencia')
plt.show()

# Comparación de precisión (bias y varianza)
print(f'Precisión en el conjunto de entrenamiento: {train_accuracy:.2f}')
print(f'Precisión en el conjunto de prueba: {accuracy:.2f}')

# Conclusión sobre el ajuste del modelo
if train_accuracy < 0.8 and accuracy < 0.8:
    print("El modelo está subajustado (underfitting).")
elif train_accuracy > 0.9 and accuracy < 0.8:
    print("El modelo está sobreajustado (overfitting).")
else:
    print("El modelo tiene un ajuste adecuado.")
