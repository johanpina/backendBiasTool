# Changelog — Backend Sesgos y Equidad

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/).

## [Fase 3.1] — EDA ampliado e interactivo (pestañas)

### Added
- **Backend `/api/eda`** enriquecido: histogramas para variables numéricas,
  índice de **balance** (entropía normalizada) por categórica, **role_hint**
  (resultado / protegida / identificador / atributo), y **tablas de contingencia
  precomputadas** entre columnas categóricas (para vistas bivariadas sin re-subir
  el CSV).
- **EDA reorganizado en 4 pestañas** en el frontend:
  - **Resumen**: stat tiles, alertas (con "explorar →" para proxies) y roles sugeridos.
  - **Distribuciones**: explorador interactivo (barra/histograma, toggle conteo/%,
    badge de balance) + grid de perfil de columnas.
  - **Relaciones**: heatmap de asociación **clickable** + explorador bivariado
    (barras agrupadas por conteo o apiladas 100% por proporción).
  - **Vista de sesgo**: **resultado por grupo** (barras apiladas 100% — evidencia
    visual de disparidades) + **heatmap de intersecciones** (detecta celdas
    intersectivas diminutas, p. ej. "mujeres asiáticas").
- Tests backend (histograma, balance, role_hint, crosstabs). **47 pruebas verdes.**

### Fixed
- Barras del explorador de distribución no renderizaban con `<Cell>` en Recharts;
  se reemplazó por un `shape` personalizado (mantiene color por barra, incluido el
  ámbar para subgrupos pequeños).

## [Fase 3] — Módulo de Análisis Exploratorio (EDA)

Nuevo paso previo al análisis que evidencia **desbalances** y **proxies** de forma
visual e intuitiva, antes de medir la equidad.

### Added
- **`api/eda.py` + endpoint `POST /api/eda`**: perfila el dataset y devuelve JSON:
  - Perfil por columna: tipo (numérica/categórica/binaria), % de nulos,
    cardinalidad, top-valores con conteos, y estadísticas numéricas.
  - **Matriz de asociación (Cramér's V corregido)** entre variables categóricas
    para detectar **proxies** (variables que "esconden" un atributo protegido).
  - **Alertas** de calidad y sesgo: nulos altos, alta cardinalidad (IDs),
    subgrupos con muestra pequeña, categorías dominantes (desbalance), columnas
    constantes y pares fuertemente asociados (proxy).
- **Panel EDA en el frontend** (`components/eda/`, con **Recharts**):
  - Stat tiles (filas, columnas, % faltantes, nº de alertas).
  - Panel de alertas con severidad (crítica/advertencia/info), iconos y colores.
  - **Mapa de asociaciones** (heatmap Cramér's V) como detector visual de proxies.
  - Explorador de distribución interactivo y grid de perfil por columna con
    mini-distribuciones (subgrupos pequeños resaltados en ámbar).
  - Paleta de visualización validada (dataviz skill).
- `api/tests/test_eda.py`: estructura, tipos, alertas, detección de proxy,
  Cramér's V y endpoint. **45 pruebas verdes.**

### Changed
- El EDA se ejecuta automáticamente al cargar el CSV y se muestra **antes** de la
  configuración del análisis, para informar la selección de columnas.
- `recharts` fijado a `^2.15` (React 18; la 3.x arrastra react-redux y provocaba
  "Invalid hook call / more than one copy of React"). `vite.config.ts` con
  `resolve.dedupe` de React.

## [Fase 2.1] — Muestra mínima, tolerancia como multiplicador y transparencia

Resuelve la confusión de que un atributo (p. ej. `race`) "nunca cambiaba" aunque
se subiera la tolerancia.

### Added
- **Umbral de muestra mínima** (`min_group_size`, default 50) en `metrics_core.py`:
  los subgrupos con `group_size < min_group_size` se marcan `insufficient_sample`
  y **no determinan** la conclusión de equidad del atributo (siguen visibles).
  Evita que subgrupos diminutos y poco fiables (p. ej. n=18, n=32) dominen el
  veredicto. Se expone en `metadata.min_group_size` y se acepta como parámetro en
  `/api/full_analysis` y `/api/recalculate_fairness`.
- **Panel de transparencia** en el frontend ("¿Por qué estas conclusiones?"):
  por cada atributo inequitativo lista los subgrupos fiables que lo causan, la
  métrica de mayor disparidad y su valor, y los subgrupos excluidos por muestra
  insuficiente.
- Tests de muestra mínima (marca + exclusión del veredicto). **39 pruebas verdes.**

### Changed
- **La tolerancia se presenta como multiplicador `×`** (1.00×–3.00×) en vez de
  porcentaje, con la banda de equidad explícita ("equitativo si la disparidad
  está entre 1/τ y τ"). Coincide con los valores de las tablas (ratios respecto a
  1.00) y evita porcentajes poco intuitivos. Aplicado a los dos sliders
  (configuración inicial y "Recalcular").
- La columna interna `insufficient_sample` se oculta de la tabla de disparidades
  (se mostraría como un check verde engañoso) y se comunica vía el panel.

### Fixed
- `AnalysisConfiguration.tsx`: `class` → `className` (eliminaba warnings de React
  y hacía que estilos del Paso 3 no se aplicaran).

## [Fase 2] — Soporte multiclase (one-vs-rest)

### Added
- **Análisis de modelos multiclase** (>2 clases) con estrategia one-vs-rest en
  `metrics_core.py`:
  - `_detect_task` distingue binario ({0,1}) de multiclase automáticamente.
  - `_binarize_ovr` evalúa cada clase frente al resto reutilizando el núcleo
    binario (`_run_binary_tables`).
  - La respuesta multiclase añade: columna `class` en las tablas globales,
    `by_class` (tablas + gráfico por clase, en forma binaria), `fairness_overall`
    (Unfair si lo es en alguna clase) y `metadata.classes`.
- `main.py`: serialización de `by_class`/`fairness_overall` (helper
  `serialize_tables`) y traducción de la columna `class` → "Clase".
- Selector de **clase a analizar** en el frontend (`ToolView.tsx`): cuando el
  modelo es multiclase, muestra un selector one-vs-rest y un resumen global con
  badges; las pestañas de Sesgos/Equidad reutilizan las tablas de la clase
  seleccionada sin cambios internos.
- `api/tests/test_multiclass.py`: detección de tarea, estructura `by_class`,
  equivalencia OvR vs. binarización manual, `fairness_overall` conservador.
  **37 pruebas, todas verdes.**

### Fixed
- Paleta de los gráficos de distribución dimensionada al número de categorías
  (evita el warning de seaborn con 3+ clases).

### Notes
- El caso **binario permanece idéntico** (sin `by_class` ni columna `class`),
  por lo que no hay regresión para modelos de 2 clases.

## [Fase 1.1] — Paridad visual con Aequitas y equidad interactiva

### Added
- `api/plots.py`: visualizaciones estilo Aequitas en matplotlib puro:
  - `render_group_metric_plot`: barras horizontales agrupadas por atributo,
    coloreadas por tamaño de grupo, etiquetadas "VALOR (Num: N), X.XX".
  - `render_disparity_treemap`: cuadrícula de treemaps de disparidad (uno por
    atributo) con grupo de referencia "(Ref)", escala divergente centrada en 1
    (colorbar 0–2) y footnote "No etiquetado arriba" para grupos pequeños.
    Incluye un algoritmo `squarify` propio (sin dependencias nuevas).
- `api/tests/test_plots.py` y tests de regresión de equidad en
  `test_metrics_core.py`. **30 pruebas, todas verdes.**
- Endpoint `GET /api/health` (documentado en el README pero ausente en el código).

### Fixed
- **La conclusión de equidad siempre daba "No Equitativo" y no respondía a la
  tolerancia.** Dos causas:
  1. `fairness_conclusion` se basaba en la Paridad Estadística (`ppr_disparity`),
     que depende del tamaño relativo del grupo y casi siempre es Unfair. Ahora se
     basa en las métricas de error que la herramienta prioriza (**FPR, FNR, FOR,
     FDR**): un subgrupo es "Unfair" si cualquiera de ellas supera la tolerancia.
     Así la conclusión responde al umbral τ.
  2. El umbral del slider de la pestaña de Equidad se enviaba como
     `fairness_threshold` pero el backend leía `fairnessThreshold`, por lo que la
     tolerancia nunca se aplicaba. Ahora `main.py` acepta ambas claves y
     `ToolView.tsx` propaga el umbral del slider correctamente.
- Los gráficos de barras y disparidad ahora replican el aspecto de los gráficos
  originales de Aequitas (antes eran barras simples de un solo color).

### Changed
- `bias_analysis.render_absolute_plot` / `render_disparity_plot` delegan en
  `api/plots.py`. El gráfico de disparidad inicial usa `fpr_disparity`.


## [Fase 1] — Motor híbrido ligero (binario)

Reemplazo de Aequitas por un motor propio ligero, preservando exactamente el
contrato de datos que consume el frontend y el comportamiento binario.

### Added
- `api/metrics_core.py`: motor de métricas en pandas/numpy/scipy (crosstabs,
  disparidades vs. grupo de referencia, paridades Fair/Unfair). Reproduce las
  12 métricas absolutas y 10 disparidades de Aequitas.
- Soporte de las 4 estrategias de grupo de referencia: `majority`, `minority`,
  `custom` y `best_performance` (esta última con referencia por-métrica, igual
  que `Bias.get_disparity_min_metric` de Aequitas).
- Suite de pruebas `api/tests/`:
  - **Golden COMPAS**: reproduce exactamente los números de Aequitas 1.0.0
    (fixtures en `api/tests/fixtures/golden_*.csv`).
  - **Validación cruzada con Fairlearn** (`MetricFrame`) de las tasas por grupo.
  - Estrategias de referencia, identidades, contrato de columnas, idempotencia
    y smoke tests de todos los endpoints. **20 pruebas, todas verdes.**
- `requirements-dev.txt` (pytest, httpx, fairlearn) y `pytest.ini`.
- `to_records()` en `main.py`: convierte `NaN → None` para garantizar JSON
  parseable por el navegador con cualquier dataset (subgrupos pequeños).

### Changed
- `api/bias_analysis.py`: ahora delega el cálculo en `metrics_core` y solo
  conserva la capa de visualización. Los gráficos de disparidad/valores
  absolutos se reimplementaron con **matplotlib puro** (se migrarán a Recharts
  en el frontend en la Fase 4).
- `Dockerfile`: `python:3.9-bullseye` + `build-essential` → **`python:3.12-slim`
  sin herramientas de compilación**. Se quitó `--reload` (solo para desarrollo).

### Removed
- Dependencia **Aequitas 1.0.0** (y su cadena `fairgbm`/`lightgbm`). Nota: esta
  versión requería `Python < 3.12`, lo que bloqueaba modernizar el runtime.
- Monkeypatch de `fairgbm` en `bias_analysis.py`.
- Código muerto: `api/fairness.py`, `api/plot_aequitas_es.py`.
- `vercel.json`: despliegue unificado en Docker slim.

### Notas de compatibilidad
- La estructura de la respuesta de `/api/full_analysis` es idéntica; el frontend
  no requiere cambios en esta fase.
- Se añadió el campo aditivo `metadata.task_type` (`"binary"`), base para el
  soporte multiclase de la Fase 2.
