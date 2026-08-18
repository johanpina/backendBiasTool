# Flujo del Análisis de Sesgos en el Notebook

> **Nota (Fase 1).** El backend ya **no usa Aequitas**: el cálculo lo realiza el
> motor propio `api/metrics_core.py` (pandas/numpy/scipy). El flujo conceptual de
> tres fases descrito abajo (Group → Bias → Fairness) **sigue siendo exacto** —
> es la definición de las métricas que el nuevo motor reproduce fielmente
> (validado contra Aequitas real y Fairlearn). El notebook se conserva como
> referencia histórica y fuente del golden test.

El flujo de análisis de sesgos sigue una secuencia lógica y estructurada en tres fases principales.

> **Fase 3 — EDA.** Antes de las fases de abajo, la herramienta ejecuta un
> **Análisis Exploratorio de los Datos** (`/api/eda`): perfila columnas, calcula
> la asociación entre variables (Cramér's V) para detectar **proxies**, y emite
> alertas de desbalance/calidad. Es un paso previo que ayuda a elegir bien las
> columnas y a anticipar sesgos.

Aquí se describe el proceso paso a paso:

### Fase 1: Exploración y Visualización Inicial de los Datos

1.  **Carga de Datos**: Se carga el conjunto de datos (en el ejemplo, el dataset COMPAS sobre reincidencia delictiva) en un DataFrame de Pandas.
2.  **Identificación de Columnas**: Se definen las columnas clave: las **variables protegidas** (ej. `race`, `sex`, `age_cat`), la **predicción del modelo** (`score`) y el **valor real** o etiqueta verdadera (`label_value`).
3.  **Visualización de Distribuciones**: Se utilizan gráficos de barras (`countplot` de Seaborn) para visualizar cómo se distribuyen las predicciones y los valores reales entre los diferentes subgrupos de las variables protegidas. Esto ofrece una primera idea visual de posibles desequilibrios en el dataset y en los resultados del modelo.

### Fase 2: Análisis con el Trío de Clases de Aequitas

Aequitas divide el análisis en tres pasos secuenciales, cada uno con su propia clase de Python.

#### Paso 2.1: `Group()` - ¿Cómo se desempeña el modelo en cada grupo?

El objetivo aquí es calcular las métricas de rendimiento del modelo para cada subgrupo demográfico de forma aislada.

-   **Cálculo**: Se utiliza la clase `Group()` para procesar el DataFrame.
-   **Resultados**: Genera una tabla (`crosstab`) que contiene:
    -   **Conteos brutos**: Falsos positivos (FP), falsos negativos (FN), verdaderos positivos (TP) y verdaderos negativos (TN) para cada grupo (ej. para 'raza: Caucásico', 'raza: Afroamericano', etc.).
    -   **Métricas absolutas**: A partir de los conteos, calcula métricas de error y rendimiento como la Tasa de Falsos Positivos (FPR), Tasa de Verdaderos Positivos (TPR), Precisión, etc., para cada uno de esos mismos grupos.

En esta fase, solo se observa el rendimiento de cada grupo sin compararlos entre sí.

#### Paso 2.2: `Bias()` - ¿Existen disparidades entre los grupos?

Una vez que se sabe cómo rinde el modelo en cada grupo, el siguiente paso es compararlos para cuantificar las disparidades.

-   **Cálculo**: Se usa la clase `Bias()`, que toma como entrada la tabla generada por `Group()`.
-   **Grupo de Referencia**: Se elige un **grupo de referencia** con el cual se compararán todos los demás. El notebook muestra que esto se puede hacer de tres maneras:
    1.  El usuario lo define manually (ej. `race: Caucasian`).
    2.  Se elige automáticamente el grupo mayoritario.
    3.  Se elige automáticamente el grupo con el "mejor" rendimiento en una métrica específica.
-   **Resultados**: Genera una tabla de **disparidades**. Cada métrica de disparidad se calcula como el cociente entre la métrica del grupo y la del grupo de referencia (ej. `Disparidad_FPR = FPR_grupo_X / FPR_grupo_referencia`). Un valor de 1.0 indica que no hay disparidad.

#### Paso 2.3: `Fairness()` - ¿Son estas disparidades justas?

El último paso es interpretar si las disparidades calculadas son aceptables o no.

-   **Cálculo**: Se utiliza la clase `Fairness()`, que toma como entrada la tabla de disparidades de `Bias()`.
-   **Umbral de Equidad (`tau`)**: Se define un umbral de tolerancia (generalmente 1.25, lo que significa que se tolera una disparidad de hasta un 25%).
-   **Resultados**: Genera una tabla final que añade una columna de `fairness_conclusion`. Compara cada valor de disparidad con el umbral y lo etiqueta como **"Fair"** (justo) si está dentro del umbral o **"Unfair"** (injusto) si lo supera. También se genera un gráfico resumen que visualiza estas conclusiones de forma global.

En resumen, el flujo es un proceso de profundización: se empieza con una visión general de los datos, luego se calculan métricas de rendimiento para cada grupo de forma aislada (`Group`), después se cuantifican las diferencias entre ellos (`Bias`), y finalmente se emite un juicio sobre si esas diferencias son problemáticas (`Fairness`).
