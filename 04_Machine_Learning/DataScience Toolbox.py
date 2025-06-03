# 📈 Visualización de datos y resultados

import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions_vs_actual(y_real, y_pred):
    """
    Gráfico de valores reales vs predichos en problemas de regresión.
    Ideal para evaluar visualmente el ajuste del modelo.

    Args:
        y_real (array-like): Valores reales.
        y_pred (array-like): Valores predichos.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y_real, alpha=0.5)
    plt.xlabel("Valores Predichos")
    plt.ylabel("Valores Reales")

    max_value = max(max(y_real), max(y_pred))
    min_value = min(min(y_real), min(y_pred))
    plt.plot([min_value, max_value], [min_value, max_value], 'r')

    plt.title("Comparación de Valores Reales vs. Predichos")
    plt.show()


# 📊 Métricas de evaluación (clasificación)

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def mostrar_reporte_clasificacion(y_true, y_pred):
    """
    Muestra el informe de clasificación y la matriz de confusión normalizada por clase.
    """
    print(classification_report(y_true, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='true')


# ⚖️ Técnicas de balanceo de clases

from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

def aplicar_smote(X_train, y_train, random_state=42):
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)

def aplicar_undersampling(X_train, y_train, clase_mayoritaria='no', clase_minoritaria='yes'):
    may = X_train[y_train == clase_mayoritaria]
    min_ = X_train[y_train == clase_minoritaria]

    may_down = resample(may, replace=False, n_samples=len(min_), random_state=42)

    X_bal = pd.concat([may_down, min_])
    y_bal = pd.concat([y_train.loc[may_down.index], y_train.loc[min_.index]])
    return X_bal, y_bal

# 🧼 Preprocesamiento y transformación

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def escalar_minmax(df, columnas):
    scaler = MinMaxScaler()
    df[columnas] = scaler.fit_transform(df[columnas])
    return df, scaler

def escalar_estandar(df, columnas):
    scaler = StandardScaler()
    df[columnas] = scaler.fit_transform(df[columnas])
    return df, scaler


# 🔁 Validación y optimización de modelos

from sklearn.model_selection import cross_val_score, GridSearchCV

def buscar_mejor_k_knn(X, y, k_range=range(1, 21)):
    from sklearn.neighbors import KNeighborsClassifier
    metricas = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(model, X, y, cv=5, scoring="balanced_accuracy").mean()
        metricas.append(score)
    best_k = k_range[np.argmax(metricas)]
    return best_k, metricas

# 📦 Utilidades varias

def aplicar_get_dummies(df, columnas_categoricas):
    return pd.get_dummies(df, columns=columnas_categoricas, dtype=int)

def convertir_ordinal(df, columna, mapeo):
    return df[columna].map(mapeo)

# 🔍 Análisis de errores en regresión
residuos = y_test - y_pred
# Gráfico de distribución de residuos
def plot_residual_distribution(residuos):
    """
    Muestra la distribución de los residuos (errores de predicción) en una regresión.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(residuos, kde=True)
    plt.title('Distribución de Residuos')
    plt.xlabel('Error de Predicción')
    plt.ylabel('Frecuencia')
    plt.show()

# Gráfico de residuos vs. predicciones
def plot_residuals_vs_predictions(y_pred, residuos):
    """
    Muestra un gráfico de dispersión de residuos vs predicciones para detectar sesgos.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuos, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuos vs. Predicciones')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    plt.show()








