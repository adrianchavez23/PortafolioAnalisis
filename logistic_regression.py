import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns

df = pd.read_csv("heart-disease.csv")

df.columns = df.columns.str.strip()

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
label = ['target']

df_X = df[features]
df_y = df[label]


# colum bool -> int
for col in df_X.columns:
    if df_X[col].dtype == 'bool':
        df_X[col] = df_X[col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.25, random_state=42)

# Escalar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# entrenar KNN
regression = LogisticRegression()
regression.fit(X_train, y_train.values.ravel())  

# predicciones
y_pred = regression.predict(X_test)

# Evaluación
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')

# matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=regression.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.show()

# Reporte de clasificación
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Gráfica de la distribución de las probabilidades predichas
probs = regression.predict_proba(X_test)[:, 1]  
sns.histplot(probs, kde=True, bins=10)
plt.title('Distribución de las Probabilidades Predichas')
plt.xlabel('Probabilidad Predicha')
plt.ylabel('Frecuencia')
plt.show()