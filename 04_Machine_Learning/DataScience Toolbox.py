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



# 📈 Visualización de datos y resultados

import pandas as p
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal, ttest_ind
from scipy.stats import pearsonr

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

def apply_log_transformation(train_set, test_set, features_to_transform):
    """
    Aplica la transformación logarítmica a columnas numéricas,
    asegurando que todos los valores sean positivos para evitar errores con log().
    Se desplazan los valores si es necesario.

    Args:
        train_set (DataFrame): Conjunto de entrenamiento.
        test_set (DataFrame): Conjunto de test.
        features_to_transform (list): Lista de nombres de columnas a transformar.

    Returns:
        Tuple[DataFrame, DataFrame]: Conjuntos de datos transformados.
    """
    train_set_transformed = train_set.copy()
    test_set_transformed = test_set.copy()

    for col in features_to_transform:
        desplaza = 0
        if train_set_transformed[col].min() <= 0:
            desplaza = int(abs(train_set_transformed[col].min())) + 1
        train_set_transformed[col] = np.log(train_set_transformed[col] + desplaza)
        test_set_transformed[col] = np.log(test_set_transformed[col] + desplaza)
    
    return train_set_transformed, test_set_transformed


def plot_target_distribution(df, target):
    """
    Grafica la distribución de frecuencias de la variable target.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        target (str): Nombre de la variable objetivo.
    """
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=target, hue=target, palette="pastel", legend=False)
    plt.title(f"Distribución de la variable objetivo: {target}")
    plt.xlabel(target)
    plt.ylabel("Frecuencia")
    total = len(df)
    
    for p in plt.gca().patches:
        height = p.get_height()
        pct = height / total * 100
        plt.text(p.get_x() + p.get_width()/2, height + 100, f'{pct:.1f}%', ha="center")

    plt.tight_layout()
    plt.show()

def plot_target_distribution_continuous(df, target, bins=30, kde=True, color="skyblue"):
    """
    Grafica la distribución de una variable continua con histograma y KDE opcional.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        target (str): Nombre de la variable continua a graficar.
        bins (int): Número de bins para el histograma.
        kde (bool): Si True, dibuja la línea KDE.
        color (str): Color del histograma.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=target, bins=bins, kde=kde, color=color)
    plt.title(f"Distribución de la variable continua: {target}")
    plt.xlabel(target)
    plt.ylabel("Frecuencia")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def escalar_minmax(df, columnas):
    scaler = MinMaxScaler()
    df[columnas] = scaler.fit_transform(df[columnas])
    return df, scaler

def escalar_estandar(df, columnas):
    scaler = StandardScaler()
    df[columnas] = scaler.fit_transform(df[columnas])
    return df, scaler

def descripcion_variables(df, umbral_categoria=10):
    """
    Muestra una descripción textual de las variables de un DataFrame,
    clasificándolas como binarias, categóricas, discretas o continuas.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        umbral_categoria (int): Número máximo de valores únicos para que una variable numérica discreta
                                sea considerada categórica.
    """
    for col in df.columns:
        print(f"\n🔹 Variable: {col}")
        tipo = df[col].dtype
        print(f"Tipo: {tipo}")

        n_unicos = df[col].nunique()
        valores_unicos = df[col].dropna().unique()

        if n_unicos == 2:
            print("Tipo inferido: Binaria")
            modo = df[col].mode().iloc[0]
            frecuencia = df[col].value_counts().iloc[0]
            print(f"Más frecuente: {modo} ({frecuencia} veces)")

        elif tipo == "object":
            print("Tipo inferido: Categórica")
            print(f"Valores únicos: {n_unicos}")
            print(f"Más frecuente: {df[col].mode().iloc[0]} ({df[col].value_counts().iloc[0]} veces)")

        elif pd.api.types.is_integer_dtype(df[col]) and n_unicos <= umbral_categoria:
            print("Tipo inferido: Discreta (recuento entero con pocos valores únicos)")
            print(f"Valores únicos: {n_unicos}")
            print(f"Más frecuente: {df[col].mode().iloc[0]} ({df[col].value_counts().iloc[0]} veces)")

        elif pd.api.types.is_numeric_dtype(df[col]):
            print("Tipo inferido: Numérica continua")
            print(f"Media: {df[col].mean():.2f}")
            print(f"Desviación estándar: {df[col].std():.2f}")
            print(f"Mínimo: {df[col].min()}")
            print(f"Máximo: {df[col].max()}")

        else:
            print("Tipo no identificado con claridad.")

def describe_df(df):
    """
    Recibe un dataframe y devuelve otro dataframe con los nombres 
    de las columnas del dataframe transferido a la función. En filas contiene 
    los parámetros descriptivos del dataframe: 
    - Tipo de dato
    - % Nulos (Porcentaje de valores nulos)
    - Valores Únicos
    - % Cardinalidad (Relación de valores únicos con el total de registros)
    
    Argumentos:
    - df (DataFrame): DataFrame de trabajo
    
    Retorna:
    - DataFrame con los parámetros descriptivos
    """
    
    summary_df = pd.DataFrame(index=['Tipo', '% Nulos', "Valores infinitos", 'Valores Únicos', '% Cardinalidad'])

    for column in df.columns:
        tipo = df[column].dtype
        porcentaje_nulos = df[column].isnull().mean() * 100
        verificar_si_es_numerico = np.issubdtype(df[column].dtype,np.number)
        if (verificar_si_es_numerico):
            valores_inf = ("Yes" if np.isinf(df[column]).any() else "No")
        else:
            valores_inf = "No"
        valores_unicos = df[column].nunique()
        cardinalidad = (valores_unicos / len(df)) * 100

        summary_df[column] = [tipo, f"{porcentaje_nulos:.2f}%",valores_inf, valores_unicos, f"{cardinalidad:.2f}%"]

    return summary_df


def tipifica_variables(df, umbral_categoria=10, umbral_continua=0.2):
    """
    Clasifica las columnas de un DataFrame según su tipo de variable: Binaria, Categórica, Numérica Discreta o Continua.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    umbral_categoria (int): Máximo número de valores únicos para que una variable sea considerada categórica.
    umbral_continua (float): Porcentaje mínimo (sobre total de filas) para considerar una variable como continua.

    Devuelve:
    pd.DataFrame: DataFrame con columnas 'nombre_variable' y 'tipo_sugerido'.
    """
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

def get_features_num_regression(df,target_col,umbral_corr,pvalue=None,mostrar=False):
    """
    Conseguir la lista de features que tienen gran impacto en la target.
    Argumentos:
        df:DataFrame que se pasa de entrada
        target_col:la variable target con el que se quiere analizar,tiene que ser numerico
        umbral_corr: Umbral minimo para considerar una variable importante para el modelo
        pvalue: Certeza estadística con la que queremos validar la importancia de las feature
    Returns:
        Lista:Lista de features importantes.
        Mostrar: Muestra la matriz de correlación en una grafica HeatMap.
    """
    if target_col not in df.columns:
        raise ValueError(f"Columna target {target_col} no esta en el DataFrame dado.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El dato de entrada tiene que ser un DataFrame.")
    
    if umbral_corr < 0 or umbral_corr > 1:
        raise ValueError("Umbral de correlacion tiene que estar entre 0 y 1")
    if pvalue is not None and (pvalue < 0 or pvalue > 1):
        raise ValueError("P-value tiene que estar entre 0 y 1")

    if not (np.issubdtype(df[target_col].dtype,np.number)):
        raise TypeError(f"Columna target {target_col} tiene que ser numerico amigo")
    
    cardinalidad_target = df[target_col].nunique() / len(df) * 100
    if cardinalidad_target < 10:
        warnings.warn(f"Columna target {target_col} tiene poca cardinalidad ({cardinalidad_target:.2f}%).")
    if cardinalidad_target > 95:
        warnings.warn(f"Columna target {target_col} tiene mucha cardinalidad ({cardinalidad_target:.2f}%).")
    if cardinalidad_target == 100:
        raise ValueError(f"Columna target {target_col} tiene 100% cardinalidad.")

    if df[target_col].isnull().sum() > 0:
        raise ValueError(f"Columna target {target_col} tiene valores Nulos.")
    
    corr = df.corr(numeric_only=True)[target_col]
    if mostrar:
        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(numeric_only=True),annot=True, cmap='coolwarm', center=0)
        plt.title(f"Correlation heatmap for {target_col}")
        plt.show()
    
    corr = corr[abs(corr) > umbral_corr]
    corr = corr.drop(target_col)
    lista = []
    if pvalue is not None:
        pvalues = []
        for col in corr.index:
            _, p = pearsonr(df[target_col], df[col])
            if p < pvalue:
                lista.append(col)
    return lista

from scipy.stats import pointbiserialr
import warnings

def get_features_num_classification(df, target_col, umbral_corr=0.1, pvalue=None, mostrar=False):
    """
    Devuelve las variables numéricas más correlacionadas con una variable target binaria.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        target_col (str): Nombre de la columna objetivo (debe ser binaria).
        umbral_corr (float): Valor mínimo absoluto de correlación para seleccionar una variable.
        pvalue (float or None): Nivel de significancia para filtrar correlaciones (si se desea).
        mostrar (bool): Si True, muestra un heatmap de correlaciones con la target.

    Returns:
        list: Lista de columnas numéricas que cumplen los criterios de selección.
    """

    if target_col not in df.columns:
        raise ValueError(f"La columna {target_col} no está en el DataFrame.")

    if df[target_col].nunique() != 2:
        raise ValueError("La variable target debe ser binaria (solo dos valores únicos).")

    if df[target_col].isnull().sum() > 0:
        raise ValueError("La variable target contiene valores nulos.")

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.drop(target_col)
    resultados = []

    for col in num_cols:
        if df[col].isnull().sum() == 0:
            corr, p = pointbiserialr(df[target_col], df[col])
            if abs(corr) >= umbral_corr and (pvalue is None or p < pvalue):
                resultados.append((col, corr, p))
        else:
            continue  # Saltar columnas con nulos

    resultados_ordenados = sorted(resultados, key=lambda x: abs(x[1]), reverse=True)
    
    if mostrar:
        corr_map = df[[target_col] + [r[0] for r in resultados_ordenados]].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_map, annot=True, cmap="coolwarm", center=0)
        plt.title(f"Correlaciones con {target_col}")
        plt.show()

    return [r[0] for r in resultados_ordenados]

from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_most_correlated_features(df, target_col, umbral_corr=0.1, pvalue=None, mostrar=False, top_n=None):
    """
    Devuelve las variables numéricas más correlacionadas con la variable target usando Pearson.

    Args:
        df (pd.DataFrame): DataFrame con los datos (todas las columnas deben ser numéricas).
        target_col (str): Nombre de la variable objetivo.
        umbral_corr (float): Umbral mínimo de correlación absoluta para mostrar resultados.
        pvalue (float or None): Si se indica, filtra por significancia estadística.
        mostrar (bool): Si True, muestra un heatmap de correlaciones con la target.
        top_n (int or None): Número máximo de variables a devolver (por orden de correlación).

    Returns:
        list: Lista de nombres de columnas numéricas más correlacionadas con el target.
    """

    if target_col not in df.columns:
        raise ValueError(f"La columna {target_col} no está en el DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError("La variable target debe ser numérica.")

    if df[target_col].isnull().sum() > 0:
        raise ValueError("La variable target contiene valores nulos.")

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.drop(target_col)
    resultados = []

    for col in num_cols:
        if df[col].isnull().sum() == 0:
            corr, p = pearsonr(df[target_col], df[col])
            if abs(corr) >= umbral_corr and (pvalue is None or p < pvalue):
                resultados.append((col, corr, p))

    resultados_ordenados = sorted(resultados, key=lambda x: abs(x[1]), reverse=True)

    if top_n:
        resultados_ordenados = resultados_ordenados[:top_n]

    if mostrar:
        columnas_corr = [target_col] + [r[0] for r in resultados_ordenados]
        sns.heatmap(df[columnas_corr].corr(), annot=True, cmap="coolwarm", center=0)
        plt.title(f"Heatmap de correlaciones con {target_col}")
        plt.show()

    return [r[0] for r in resultados_ordenados]


def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Genera pairplots con variables numéricas del dataframe que cumplan ciertas condiciones de correlación con target_col.

    Parámetros:
    - df (DataFrame): DataFrame de entrada.
    - target_col (str): Columna objetivo para el análisis de correlación.
    - columns (list): Lista de columnas a considerar; si está vacía, se tomarán todas las numéricas del dataframe.
    - umbral_corr (float): Umbral mínimo absoluto de correlación para incluir variables.
    - pvalue (float or None): Nivel de significación estadística para el test de correlación. Si es None, no se aplica.

    Retorna:
    - Lista de columnas seleccionadas según las condiciones.
    """

    # Validación de target_col (debe existir enel dataframe)
    if target_col not in df.columns:
        raise ValueError(f"La columna target_col '{target_col}' no existe en el dataframe.")

    # Si columns está vacío, tomamos todas las variables numéricas excepto target_col
    if not columns:
        columns = df.select_dtypes(include=np.number).columns.tolist() #añado todas las columnas numéricas a la lista de columnas vacía
        columns.remove(target_col) #le quito el target
    #si hay columnas en parámetros tomará esas

    # Filtrar columnas por correlación
    selected_columns = [] #creo una lista vacía de columnas seleccionadas que iré rellenando
    for col in columns: #recorro las columnas de columns
        if col == target_col: #si la col es el target me la salto
            continue
        corr = df[[target_col, col]].dropna().corr().iloc[0, 1]  # tomo las columnas target_col y col del dataframe, elimina los NaN y calcula la matriz de correlación y extrae el valor con el iloc
        
        if abs(corr) > umbral_corr: #si la correlación en valor absoluto es mayor del umbral
            # Si se especifica pvalue, verificar significación estadística
            if pvalue is not None: #si el pvalue no es None
                _, pval = pearsonr(df[target_col].dropna(), df[col].dropna()) #calculo la correlación entre target_col y col y devuelve el vp_val porque el corr_coef no me hace falta
                if pval < 1 - pvalue: #si la probabilidad pval de que la correlación ocurra al azar es menor de 1-pvalue
                    selected_columns.append(col) #es estadísticamente signifcativa y lo meto en la lista
            else:
                selected_columns.append(col) # si no hay pvalue agrega la columna a la lista para verificarla

    # Graficar en grupos de máximo 5 columnas por gráfico
    if selected_columns: #si selected_columns no está vacía
        for i in range(0, len(selected_columns), 4):  # Genero números e 0 a la longitud de selected_columns de 4 en 4. Máximo 5 con target_col, proceso 4 columnas de cada iteración
            subset = [target_col] + selected_columns[i:i+4] #creo este subset que tiene el target y las 4 columnas
            sns.pairplot(df[subset].dropna(), diag_kind='kde') #hago el pairplot habiendo eliminado las filas con Nan con el dropna
            plt.show() #lo muestro

    return selected_columns #devuelvo las columnas que superaron el filtro de correlación y significancia

# Ejemplo de uso:
data = {
    'target': [1, 2, 3, 4, 5, 6, 7],
    'A': [2, 4, 6, 8, 10, 12, 14],
    'B': [1, 3, 3, 5, 5, 7, 7],
    'C': [5, 4, 3, 2, 1, 0, -1],
    'D': [10, 20, 30, 40, 50, 60, 70]
}
df = pd.DataFrame(data)

result = plot_features_num_regression(df, target_col="target", umbral_corr=0.5, pvalue=0.05)
print("Columnas seleccionadas:", result)

def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    Devuelve una lista de columnas categóricas que presentan una relación significativa
    con la variable numérica target_col usando t-test o ANOVA según corresponda.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    target_col (str): Nombre de la columna objetivo (numérica continua o discreta con alta cardinalidad).
    pvalue (float): Nivel de significación estadística (default = 0.05).

    Retorna:
    list or None: Lista de variables categóricas relacionadas, o None si hay error en los argumentos.
    """
    
    # Validaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("❌ 'df' debe ser un DataFrame.")
        return None

    if target_col not in df.columns:
        print(f"❌ La columna '{target_col}' no está en el DataFrame.")
        return None

    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"❌ La columna '{target_col}' no es numérica.")
        return None

    if not (0 < pvalue < 1):
        print("❌ 'pvalue' debe estar entre 0 y 1.")
        return None

    cardinalidad = df[target_col].nunique()
    porcentaje = cardinalidad / len(df)

    if cardinalidad < 10 or porcentaje < 0.05:
        print(f"❌ La variable '{target_col}' no tiene suficiente cardinalidad para considerarse continua.")
        print(f"Cardinalidad única: {cardinalidad} ({round(porcentaje * 100, 2)}%)")
        return None

    # Selección de columnas categóricas
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_cols:
        print("⚠️ No hay variables categóricas en el DataFrame.")
        return []

    relacionadas = []

    for col in cat_cols:
        niveles = df[col].dropna().unique()
        grupos = [df[df[col] == nivel][target_col].dropna() for nivel in niveles]

        if any(len(grupo) < 2 for grupo in grupos):
            continue  # no hay suficientes datos en alguno de los grupos

        try:
            if len(niveles) == 2:
                stat, p = ttest_ind(*grupos)
            elif len(niveles) > 2:
                stat, p = f_oneway(*grupos)
            else:
                continue

            if p < pvalue:
                relacionadas.append(col)
        except Exception as e:
            print(f"⚠️ Error evaluando la columna '{col}': {e}")
            continue

    return relacionadas

def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):

    """
    Identifica variables categóricas que tienen una relación significativa con una variable
    numérica continua usando ANOVA de una vía. Opcionalmente, genera histogramas agrupados.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos.

    target_col : str
        Nombre de la columna numérica continua a predecir.

    columns : list of str, opcional
        Columnas categóricas a evaluar. Si está vacío, se detectan automáticamente.

    pvalue : float, opcional
        Umbral de significancia (por defecto 0.05).

    with_individual_plot : bool, opcional
        Si es True, se grafican los histogramas por categoría.

    Retorna
    -------
    list of str
    Columnas categóricas significativamente relacionadas con la variable objetivo.

    """
    
    # Validación de DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un DataFrame.")
        return None
    
    # Validación de target_col
    if not target_col or target_col not in df.columns:
        print("Error: target_col no está en el DataFrame o es vacío.")
        return None
    
    # Validación de tipo de target_col
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser una variable numérica continua.")
        return None

    # Validación de columns
    if not isinstance(columns, list):
        print("Error: columns debe ser una lista de strings.")
        return None
    
    # Si columns está vacío, seleccionamos categóricas automáticamente
    if not columns:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    columnas_significativas = []

    for col in columns:
        if col not in df.columns:
            print(f"Aviso: La columna '{col}' no está en el DataFrame. Se omite.")
            continue

        if df[col].nunique() <= 1:
            continue  

        try:
            grupos = [df[df[col] == cat][target_col].dropna() for cat in df[col].dropna().unique()]
            if any(len(g) == 0 for g in grupos):
                continue

            f_stat, p_val = f_oneway(*grupos)

            if p_val < pvalue:
                columnas_significativas.append(col)

                if with_individual_plot:
                    plt.figure(figsize=(8, 4))
                    sns.histplot(data=df, x=target_col, hue=col, multiple="stack", kde=False)
                    plt.title(f"{col} vs {target_col} (p = {p_val:.4f})")
                    plt.tight_layout()
                    plt.show()

        except Exception as e:
            print(f"Error evaluando la columna '{col}': {e}")

    return columnas_significativas


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








