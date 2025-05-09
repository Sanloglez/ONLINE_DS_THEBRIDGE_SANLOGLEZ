# 📊 EDA – Hábitos de Juego en Videojuegos 🎮

Un proyecto de análisis exploratorio de datos (EDA) centrado en los hábitos de los jugadores: cuánto juegan, por qué juegan, qué dispositivos prefieren y qué géneros de juegos consumen más. Basado en una encuesta a 500 jugadores de videojuegos.

---

## 📁 Repositorio

```
EDA_Hábitos_de_Juego/
├── src/
│ ├── data/ # Datos procesados
│ ├── img/ # Visualizaciones exportadas
│ ├── notebooks/ # Notebooks iniciales o de prueba
│ └── utils/
│ └── utils.py # Funciones reutilizables
├── main.ipynb # Notebook principal del proyecto
├── Memoria.ipynb # Informe técnico en notebook
├── Memoria.pdf # Informe técnico en PDF
├── Presentación.pdf # Resumen visual
└── README.md # Descripción general del proyecto
```

---

## 👩‍💻 Proyecto

### Análisis de hábitos y perfiles de jugadores a partir de una encuesta
**Tecnologías:** Python, Pandas, Matplotlib, Seaborn, Jupyter Notebook  
**Keywords:** EDA, Videojuegos, Consolas, Hábitos de juego, PC, Segmentación de jugadores, Python, Jupyter, Pandas

#### Objetivos:
- Identificar qué factores influyen en el número de horas jugadas.
- Detectar diferencias de comportamiento por edad o género.
- Perfilar tipos de jugadores según dispositivo, motivación y género favorito.

#### Resumen del análisis:
- Transformación de variables categóricas múltiples en formato binario.
- Conversión de escalas ordinales a valores numéricos aproximados.
- Análisis univariante, bivariante y multivariante.
- Uso de gráficos `countplot`, `catplot`, `boxplot`, `heatmap` y `pairplot`.
- Segmentaciones por edad, género, dispositivo, motivación y género de juego.

#### Conclusiones destacadas:
Hipótesis 1: ¿Qué factores influyen más en el número de horas jugadas?

- El dispositivo y las motivaciones personales tienen un impacto claro en las horas jugadas.
- En el juego nº1, se observa que los jugadores que usan móvil tienden a jugar más horas, probablemente por su accesibilidad.
- Las motivaciones como la diversión, la competitividad y el alivio del estrés se relacionan directamente con un mayor tiempo de juego, especialmente en usuarios que superan las 10h semanales.
- Sin embargo, en el juego nº2, los factores son más difusos. No hay un dispositivo claramente dominante y las motivaciones están más repartidas, lo que sugiere que otros elementos contextuales o de diseño del juego pueden influir más que las variables estudiadas.

Hipótesis 2: ¿Hay diferencias claras según género o edad?

Sí existen diferencias significativas:
- En cuanto a edad, los jugadores jóvenes (16-20 años) prefieren dispositivos móviles y motivaciones como socializar y aliviar el estrés, mientras que los de mayor edad se enfocan más en la competitividad y la narrativa.
- Por género, las mujeres muestran una mayor inclinación hacia el alivio del estrés y la diversión, mientras que los hombres se reparten más entre socializar, historia y aprendizaje. Además, hay diferencias en dispositivos y géneros de videojuegos preferidos.

---



## 📫 Contacto

Puedes escribirme a través de [LinkedIn](https://www.linkedin.com/in/sandralopezglez89/) si estás interesado en perfiles con enfoque en análisis de datos, storytelling con datos o visualización en Python.