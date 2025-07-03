# Predicción de precios y detección de oportunidades en Airbnb Madrid

Este proyecto de Machine Learning se centra en la predicción del precio de un alojamiento de Airbnb en Madrid a partir de sus características (zona, tipo, disponibilidad, número de noches, etc.).  
Además, se identifican alojamientos cuyo precio real está significativamente por debajo del estimado por el modelo, lo que permite detectar posibles oportunidades de inversión o de ajuste de precios.

---
## Estructura del repositorio

```
Project_Break_II_ML_Airbnb_investment/
├── main.ipynb                # Notebook final con el proceso completo
├── README.md                 # Descripción general del proyecto
├── presentacion.pdf          # Resumen visual para exposición
└── src/
    ├── data/                 # Archivo geojson con mapa de Madrid
    ├── data_sample/          # Subconjunto del dataset (X_train, y_train, oportunidades, etc.)
    ├── img/                  # Imágenes exportadas 
    ├── models/               # Modelos entrenados (.joblib)
    └── utils/                # Funciones auxiliares (toolbox)
```

---
## Descripción del proyecto

Este proyecto tiene dos objetivos:

1. Construir un modelo de regresión que prediga el precio estimado de una propiedad de Airbnb en Madrid.
2. Detectar propiedades con precios por debajo de su valor estimado, que puedan representar oportunidades de inversión.

Para ello, se ha trabajado con datos públicos del portal Inside Airbnb, aplicando un pipeline de procesamiento, modelado, validación y análisis de errores.

---

## Dataset

- Fuente: Inside Airbnb (http://insideairbnb.com)
- Localización: Madrid
- Nº registros tras limpieza: ~19.000
- Variables utilizadas: 
  - Numéricas: minimum_nights, number_of_reviews, reviews_per_month, availability_365, etc.
  - Categóricas: room_type, neighbourhood, neighbourhood_group
- Filtros aplicados:
  - Eliminación de nulos
  - Eliminación de outliers
  - Transformación logarítmica de la variable objetivo (price)

---

## Proceso de modelado

1. Exploración inicial y selección de variables
2. Codificación de variables categóricas y escalado de numéricas
3. División en train/test
4. Entrenamiento de modelos base: Random Forest y XGBoost
5. Evaluación con validación cruzada
6. Optimización de hiperparámetros (GridSearchCV)
7. Aplicación de log(price) y filtrado progresivo
8. Comparación de modelos en términos de MAE, RMSE y R²
9. Persistencia del modelo final
10. Detección de apartamentos con precio real por debajo del predicho

---

## Resultados

| Métrica       | Sin filtrar | Filtro < 500 € + log(price) |
|---------------|-------------|------------------------------|
| MAE           | 65.84 €     | 33.84 €                      |
| RMSE          | 209.08 €    | 57.38 €                      |
| R²            | -0.02       | 0.37                        |

El modelo final, entrenado sobre precios filtrados y transformados logarítmicamente, logra predecir el precio de alojamientos “normales” (por debajo de 500 €) con un error medio aceptable y mayor capacidad explicativa.

---

## Oportunidades de inversión

Se ha generado un mapa interactivo con Folium que muestra aquellos apartamentos cuyo precio real está al menos un 20 % por debajo del estimado por el modelo.  
Esto permite identificar listados con precios potencialmente infravalorados, ya sea para alertar a propietarios o para oportunidades de negocio.

Archivo exportado: `src/data_sample/oportunidades_airbnb_baratos.csv`

---

## Visualización

- Gráficos EDA sobre distribución del precio y relación con otras variables
- Mapa interactivo con precios medios por barrio
- Mapa interactivo con oportunidades detectadas por el modelo

---

## Cómo ejecutar el proyecto

1. Clonar el repositorio
2. Instalar los requisitos necesarios
3. Ejecutar el archivo `main.ipynb` desde la raíz del proyecto

---

## Contacto

Sandra López  
LinkedIn: [https://www.linkedin.com/in/sandralopezglez89/](https://www.linkedin.com/in/sandralopezglez89/)

---------------------------------------------------------------------------------------------

# Price prediction and opportunity detection in Airbnb Madrid

This Machine Learning project focuses on predicting the price of Airbnb listings in Madrid based on their features (location, type, availability, minimum nights, etc.).  
Additionally, it identifies listings whose real price is significantly lower than the estimated value predicted by the model, allowing us to detect potential investment opportunities or pricing adjustments.

---
## Repository structure

Project_Break_II_ML_Airbnb_investment/
├── main.ipynb # Final notebook with the full process  
├── README.md # Project overview  
├── presentacion.pdf # Summary presentation for the video  
└── src/  
├── data/ # GeoJSON file with the Madrid map  
├── data_sample/ # Dataset subset (X_train, y_train, opportunities, etc.)  
├── img/ # Exported images  
├── models/ # Trained models (.joblib)  
└── utils/ # Utility functions (toolbox)

---
## Project description

This project has two main goals:

1. Build a regression model to estimate the price of an Airbnb property in Madrid.
2. Detect listings with a price significantly lower than the predicted value, which could represent investment opportunities.

To achieve this, we worked with public data from Inside Airbnb and applied a pipeline that includes data processing, modeling, validation, and error analysis.

---

## Dataset

- Source: Inside Airbnb (http://insideairbnb.com)
- Location: Madrid
- Records after cleaning: ~19,000
- Selected variables:
  - Numerical: minimum_nights, number_of_reviews, reviews_per_month, availability_365, etc.
  - Categorical: room_type, neighbourhood, neighbourhood_group
- Filters applied:
  - Removal of null values
  - Removal of outliers
  - Log transformation of the target variable (`price`)

---

## Modeling process

1. Initial exploration and variable selection  
2. Encoding of categorical variables and scaling of numerical variables  
3. Train/test split  
4. Training of baseline models: Random Forest and XGBoost  
5. Evaluation using cross-validation  
6. Hyperparameter tuning (GridSearchCV)  
7. Log transformation of `price` and progressive filtering  
8. Model comparison using MAE, RMSE and R²  
9. Persistence of the final model  
10. Detection of listings priced significantly below prediction

---

## Results

| Metric         | Without filtering | Filter < 500 € + log(price) |
|----------------|-------------------|------------------------------|
| MAE            | 65.84 €           | 33.84 €                      |
| RMSE           | 209.08 €          | 57.38 €                      |
| R²             | -0.02             | 0.37                         |

The final model, trained on log-transformed prices under 500 €, is able to predict prices for “standard” listings with reasonable error and improved explanatory power.

---

## Investment opportunities

An interactive map was generated using Folium to highlight listings whose real price is at least 20% lower than the model’s estimate.  
This makes it easier to identify potentially undervalued listings, either for investor targeting or owner awareness.

Exported file: `src/data_sample/oportunidades_airbnb_baratos.csv`

---

## Visualizations

- EDA plots on price distribution and variable relationships  
- Interactive map showing average prices by district  
- Interactive map showing listings detected as undervalued by the model

---

## How to run the project

1. Clone the repository  
2. Install the necessary dependencies  
3. Run `main.ipynb` from the project root

---

## Contact

Sandra López  
LinkedIn: [https://www.linkedin.com/in/sandralopezglez89/](https://www.linkedin.com/in/sandralopezglez89/)
