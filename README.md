# Backend de Análisis de Sesgo y Equidad con FastAPI

Este proyecto proporciona un backend de FastAPI para analizar el sesgo y la equidad en modelos de aprendizaje automático. El cálculo de métricas se realiza con un **motor propio ligero** (`api/metrics_core.py`, basado en pandas/numpy/scipy) que reemplaza a la antigua dependencia Aequitas, preservando las mismas métricas (crosstabs por subgrupo, disparidades vs. grupo de referencia y conclusiones de equidad Fair/Unfair). El backend procesa un archivo CSV, realiza un análisis completo y devuelve tablas de métricas y visualizaciones en un JSON estructurado.

> **Motor de métricas.** Ver `docs/DECISIONES.md` para el detalle de por qué se
> eliminó Aequitas, el contrato de columnas preservado y la semántica de los
> grupos de referencia. El motor se valida contra Aequitas real (golden test) y
> contra **Fairlearn** (`MetricFrame`) en la suite de pruebas.

## Requisitos

- Python 3.12+
- Pip

## 1. Instalación

Primero, clona el repositorio y navega al directorio del proyecto. Luego, instala las dependencias necesarias desde el archivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

Para ejecutar las pruebas, instala también las dependencias de desarrollo:

```bash
pip install -r requirements-dev.txt
pytest
```

## 2. Cómo Ejecutar el Servidor

Para iniciar el servidor de desarrollo, ejecuta el siguiente comando desde el directorio raíz del proyecto. Uvicorn cargará la aplicación FastAPI definida en `api/main.py`.

```bash
uvicorn api.main:app --reload
```

El servidor estará disponible en `http://127.0.0.1:8000`.

La opción `--reload` reiniciará el servidor automáticamente cada vez que se detecte un cambio en el código.

## 3. Documentación de la API

El backend expone un único y potente endpoint para realizar el análisis completo.

### Endpoint: `/api/full_analysis`

- **Método**: `POST`
- **Descripción**: Realiza un análisis completo de sesgo y equidad. Acepta un archivo CSV y parámetros de configuración, y devuelve un JSON con todas las tablas y gráficos resultantes.
- **Content-Type**: `multipart/form-data`

#### Parámetros de la Petición

Debido a que se envía un archivo, la petición debe ser de tipo `multipart/form-data`. Los parámetros son:

1.  `file` (requerido):
    -   **Tipo**: `UploadFile`
    -   **Descripción**: El archivo CSV que contiene los datos a analizar.

2.  `columns` (requerido):
    -   **Tipo**: `string` (JSON)
    -   **Descripción**: Un objeto JSON como string que mapea los roles de las columnas en el CSV.
    -   **Ejemplo**:
        ```json
        {
          "protected": ["race", "sex"],
          "predictions": "score",
          "actual": "label_value"
        }
        ```

3.  `params` (opcional):
    -   **Tipo**: `string` (JSON)
    -   **Descripción**: Un objeto JSON como string para configurar el análisis.
    -   **Ejemplo**:
        ```json
        {
          "referenceMethod": "majority",
          "referenceGroups": {"race": "Caucasian"},
          "fairness_threshold": 1.25
        }
        ```
    -   **Campos de `params`**:
        -   `referenceMethod`: Método para seleccionar el grupo de referencia (`majority`, `minority`, `custom`). Por defecto es `majority`.
        -   `referenceGroups`: Requerido si `referenceMethod` es `custom`. Especifica el valor del grupo de referencia para cada atributo protegido.
        -   `fairness_threshold`: Umbral numérico para determinar la equidad. Por defecto es `1.25`.

#### Estructura de la Respuesta (JSON)

El endpoint devuelve un objeto JSON con la siguiente estructura:

```json
{
  "distribution_plots": {},
  "tables": {},
  "plots": {},
  "metadata": {}
}
```

-   **`distribution_plots`**: Contiene los gráficos de distribución iniciales generados con Seaborn.
    -   Cada clave es un atributo protegido (ej. `"race"`).
    -   El valor es un objeto con `score_plot` y `label_plot`, que contienen las imágenes de los gráficos en formato **base64**.

-   **`tables`**: Contiene los 4 DataFrames principales del análisis convertidos a una lista de diccionarios (JSON).
    -   `group_counts`: Conteos de grupo (TP, FP, TN, FN, etc.).
    -   `group_metrics`: Métricas absolutas por grupo (TPR, FPR, etc.).
    -   `bias_metrics`: Métricas de disparidad comparando cada grupo con el de referencia.
    -   `fairness_summary`: El resumen final de equidad que indica si cada grupo es "Fair" o "Unfair".

-   **`plots`**: Contiene los gráficos de resumen generados por Aequitas.
    -   `disparity_summary`: Gráfico que resume todas las disparidades de métricas.
    -   `fairness_summary`: Gráfico que resume las determinaciones de equidad.
    -   Ambos gráficos son strings en formato **base64**.

-   **`metadata`**: Contiene información adicional sobre el análisis.
    -   `protected_attributes`: Lista de los atributos protegidos utilizados.
    -   `unique_values`: Diccionario con los valores únicos para cada atributo protegido.

### Endpoint de Salud

- **Método**: `GET`
- **Endpoint**: `/api/health`
- **Descripción**: Endpoint simple para verificar si la API está en funcionamiento. Devuelve `{"status": "ok"}`.
