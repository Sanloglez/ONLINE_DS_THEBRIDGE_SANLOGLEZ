# 1. Entendimiento del problema (selección de la métrica más adecuada)  
# 2. Obtención de datos y primer contacto  
# 3. Train y Test  
# 4. MiniEDA: Análisis del target, análisis bivariante, entendimiento de las features, selección de las mismas (si es necesario)  
# 5. Preparación del dataset de Train: Conversión de categóricas, tratamiento de numéricas  
# 6. Selección e instanciación de modelos. Baseline.
# 7. Comparación de modelos (lo haremos por comparación con validación, puedes hacerlo por comparación de modelos de hiperparámetros optimizados, si así lo prefieres)  
# 8. Selección de modelo: Optimización de hiperparámetros (ten en cuenta la nota de 7)  
# 9. Evaluación contra test.  
# 10. Análisis de errores, posibles acciones futuras.  
# 11. Persistencia del modelo en disco.  



# toolbox.py

# ==========================
# Librerías generales
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import pearsonr, f_oneway, ttest_ind, pointbiserialr


# ==========================
# Visualización regresión
# ==========================
def plot_predictions_vs_actual(y_real, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y_real, alpha=0.5)
    plt.xlabel("Valores Predichos")
    plt.ylabel("Valores Reales")
    max_value = max(max(y_real), max(y_pred))
    min_value = min(min(y_real), min(y_pred))
    plt.plot([min_value, max_value], [min_value, max_value], 'r')
    plt.title("Comparación de Valores Reales vs. Predichos")
    plt.show()


# ==========================
# Análisis de residuos
# ==========================
def plot_residual_distribution(residuos):
    plt.figure(figsize=(10, 6))
    sns.histplot(residuos, kde=True)
    plt.title('Distribución de Residuos')
    plt.xlabel('Error de Predicción')
    plt.ylabel('Frecuencia')
    plt.show()

def plot_residuals_vs_predictions(y_pred, residuos):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuos, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuos vs. Predicciones')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    plt.show()


# ==========================
# Escalado de datos
# ==========================
def escalar_minmax(df, columnas):
    scaler = MinMaxScaler()
    df[columnas] = scaler.fit_transform(df[columnas])
    return df, scaler

def escalar_estandar(df, columnas):
    scaler = StandardScaler()
    df[columnas] = scaler.fit_transform(df[columnas])
    return df, scaler


# ==========================
# Transformaciones
# ==========================
def apply_log_transformation(train_set, test_set, features_to_transform):
    train_set_transformed = train_set.copy()
    test_set_transformed = test_set.copy()

    for col in features_to_transform:
        desplaza = 0
        if train_set_transformed[col].min() <= 0:
            desplaza = int(abs(train_set_transformed[col].min())) + 1
        train_set_transformed[col] = np.log(train_set_transformed[col] + desplaza)
        test_set_transformed[col] = np.log(test_set_transformed[col] + desplaza)
    
    return train_set_transformed, test_set_transformed


# ==========================
# Análisis de variables
# ==========================
def describe_df(df):
    summary_df = pd.DataFrame(index=['Tipo', '% Nulos', "Valores infinitos", 'Valores Únicos', '% Cardinalidad'])

    for column in df.columns:
        tipo = df[column].dtype
        porcentaje_nulos = df[column].isnull().mean() * 100
        verificar_si_es_numerico = np.issubdtype(df[column].dtype, np.number)
        if verificar_si_es_numerico:
            valores_inf = "Yes" if np.isinf(df[column]).any() else "No"
        else:
            valores_inf = "No"
        valores_unicos = df[column].nunique()
        cardinalidad = (valores_unicos / len(df)) * 100

        summary_df[column] = [tipo, f"{porcentaje_nulos:.2f}%", valores_inf, valores_unicos, f"{cardinalidad:.2f}%"]

    return summary_df.T


def tipifica_variables(df, umbral_categoria=10, umbral_continua=0.2):
    resultado = []
    n = len(df)

    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_cardinalidad = round(cardinalidad / n, 2)

        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        elif porcentaje_cardinalidad >= umbral_continua:
            tipo = "Numerica Continua"
        else:
            tipo = "Numerica Discreta"

        resultado.append({
            "nombre_variable": col,
            "tipo_sugerido": tipo
        })

    return pd.DataFrame(resultado)


def get_features_num_regression(df, target_col, umbral_corr, pvalue=None, mostrar=False):
    if target_col not in df.columns:
        raise ValueError(f"Columna target {target_col} no está en el DataFrame.")

    corr = df.corr(numeric_only=True)[target_col]
    if mostrar:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', center=0)
        plt.title(f"Correlation heatmap for {target_col}")
        plt.show()

    corr = corr[abs(corr) > umbral_corr]
    corr = corr.drop(target_col)
    lista = []

    if pvalue is not None:
        for col in corr.index:
            _, p = pearsonr(df[target_col], df[col])
            if p < pvalue:
                lista.append(col)
    else:
        lista = list(corr.index)

    return lista


def get_features_cat_regression(df, target_col, pvalue=0.05):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    relacionadas = []

    for col in cat_cols:
        niveles = df[col].dropna().unique()
        grupos = [df[df[col] == nivel][target_col].dropna() for nivel in niveles]

        if any(len(grupo) < 2 for grupo in grupos):
            continue

        try:
            if len(niveles) == 2:
                _, p = ttest_ind(*grupos)
            elif len(niveles) > 2:
                _, p = f_oneway(*grupos)
            else:
                continue

            if p < pvalue:
                relacionadas.append(col)
        except Exception:
            continue

    return relacionadas

