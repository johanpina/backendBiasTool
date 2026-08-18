# Decisiones de arquitectura

Registro de decisiones técnicas de la modernización de la herramienta.

## D1 — Reemplazar Aequitas por un motor propio ligero (+ Fairlearn en tests)

**Contexto.** El backend usaba `aequitas==1.0.0`, que:
- Arrastraba `fairgbm`/`lightgbm` y requería `build-essential` (imagen Docker pesada).
- Necesitaba un monkeypatch de un módulo falso `fairgbm` para siquiera importarse.
- **Requiere `Python < 3.12`**, bloqueando la modernización del runtime.

**Decisión.** Reimplementar el núcleo de Aequitas (crosstabs, disparidades y
paridades) en `api/metrics_core.py` con solo `pandas`/`numpy`/`scipy`. Usar
**Fairlearn** como oráculo de validación cruzada **solo en las pruebas**, no en
la ruta caliente del request.

**Consecuencias.**
- Imagen `python:3.12-slim` sin compilación; despliegue reproducible.
- Control total sobre el cálculo (habilita multiclase en la Fase 2).
- Se asume el mantenimiento del código de métricas (mitigado por el golden test
  contra Aequitas real y la validación con Fairlearn).

## D2 — Preservar el contrato de columnas de Aequitas

Para no romper el frontend, `metrics_core` emite **los mismos nombres de columna**
que Aequitas (en inglés). Referencia congelada en `api/tests/fixtures/golden_*.csv`,
generada con Aequitas 1.0.0 real.

- **12 métricas absolutas**: `accuracy, tpr, tnr, for, fdr, fpr, fnr, npv,
  precision, ppr, pprev, prev`.
- **10 disparidades** (Aequitas NO genera `accuracy_disparity`): `ppr, pprev,
  precision, fdr, for, fpr, fnr, tpr, tnr, npv`.
- `ppr` se normaliza por el total de predichos positivos **dentro del atributo**.
- Divisiones por cero → `NaN` (no 0, no inf), salvo numerador>0 / referencia=0 → inf.

## D3 — Semántica del grupo de referencia

- `majority`/`minority`: grupo de mayor/menor `group_size`.
- `custom`: valor indicado por el usuario (error si no existe).
- `best_performance`: **referencia por-métrica** = grupo con el valor mínimo de
  cada métrica (réplica de `Bias.get_disparity_min_metric`; el parámetro
  `label_score_ref` de Aequitas solo afecta la significancia, no la referencia).

## D4 — Regla de equidad

`Fair` si la disparidad es `NaN` o cae en `[1/τ, τ]` (τ = umbral, por defecto
1.25 ≈ regla del 80%). Un valor infinito queda fuera del rango → `Unfair`.

## D5 — Base de la conclusión de equidad (`fairness_conclusion`)

**Contexto.** Originalmente `fairness_conclusion` copiaba la Paridad Estadística
(`ppr_disparity`). Como `ppr` es la *proporción* de predichos positivos del grupo
respecto al atributo, depende del tamaño relativo del grupo: cualquier grupo
minoritario tiene `ppr_disparity` muy por debajo de 1 y sale "Unfair" para casi
cualquier τ. Resultado: el resumen **siempre** daba "No Equitativo" y no
respondía a la tolerancia.

**Decisión.** La conclusión por subgrupo se basa en las métricas de **error**
que la herramienta prioriza (`FPR, FNR, FOR, FDR`, ver `FAIRNESS_CONCLUSION_METRICS`):
un subgrupo es "Unfair" si **cualquiera** de esas paridades supera τ. Son tasas
independientes del tamaño del grupo, por lo que la conclusión responde al umbral.

**Consecuencia.** El resumen de equidad es interactivo: al aumentar la tolerancia,
los atributos con disparidades de error moderadas pasan a "Equitativo". Las
paridades individuales (incluida la Estadística) siguen disponibles en la tabla
"Test de Equidad Estadística por Atributo".

## D6 — Visualizaciones sin Aequitas (paridad de aspecto)

Los gráficos se reimplementaron en matplotlib puro (`api/plots.py`) replicando el
aspecto de Aequitas (`plot_group_metric` y `plot_disparity_all`), incluido un
`squarify` propio para los treemaps. En la Fase 4 se evaluará migrarlos a Recharts
en el frontend; los treemaps son el caso más difícil de portar.

## D7 — Multiclase con one-vs-rest (OvR)

**Decisión.** Para modelos de >2 clases se evalúa cada clase **frente al resto**
(OvR), reutilizando íntegramente el núcleo binario. Cada clase produce las mismas
tablas del contrato; se exponen en `by_class[clase]` (forma binaria, para que el
frontend las consuma sin cambios) y también concatenadas con una columna `class`
(para export/consumidores de la API).

**Agregación.** `fairness_overall` marca un atributo como "Unfair" si lo es en
**cualquiera** de las clases (criterio conservador, coherente con el resumen por
atributo). El frontend muestra un selector de clase y este resumen global.

**Detección.** Binario solo si hay ≤2 valores y son numéricos {0,1}; en cualquier
otro caso (más clases, o 2 clases no numéricas como "sí"/"no") se usa OvR. El caso
binario {0,1} mantiene exactamente la salida anterior (sin `by_class`).

## D8 — Muestra mínima y presentación de la tolerancia

**Problema.** Con reglas "Unfair si algún subgrupo lo es", subgrupos diminutos
(n=18, n=32) con disparidades extremas y poco fiables forzaban un veredicto
"No Equitativo" que no cambiaba ni subiendo mucho la tolerancia. Además la
tolerancia se mostraba como porcentaje (100% = τ=2.0), lo que resultaba confuso.

**Decisiones.**
- **Muestra mínima** (`min_group_size`, default 50): los subgrupos por debajo se
  marcan `insufficient_sample` y no cuentan para la conclusión (con fallback: si
  *todos* los subgrupos de un atributo son insuficientes, se evalúan igual para no
  perder el atributo). Es práctica estándar en auditorías de sesgo.
- **Tolerancia como multiplicador `×`** (no porcentaje): mapea 1:1 con los valores
  de disparidad de las tablas (ratios respecto a 1.00) y se acompaña de la banda
  `[1/τ, τ]`.
- **Transparencia**: el frontend explica qué subgrupos fiables causan cada
  veredicto y cuáles se excluyeron por muestra insuficiente.

**Consecuencia.** Un atributo genuinamente disparejo (p. ej. `race` en COMPAS,
por Hispanic/Other, no por los grupos diminutos) se muestra como tal y el usuario
entiende *por qué*. El umbral podría hacerse configurable en la UI en la Fase 4.

## D9 — Módulo EDA orientado a sesgos

**Decisión.** Añadir un análisis exploratorio previo (`api/eda.py`, endpoint
`/api/eda`) que devuelve **solo JSON**; el frontend dibuja con **Recharts** y CSS.
Objetivo: evidenciar desbalances y **proxies** antes de medir equidad.

- **Detección de proxies** vía **Cramér's V** (corrección de Bergsma) entre
  variables categóricas: una asociación alta (≥0.5) sugiere que una variable
  puede sustituir a un atributo protegido (sesgo indirecto).
- **Alertas** accionables (muestra pequeña, alta cardinalidad, desbalance, nulos,
  proxy) coherentes con `min_group_size` del análisis de equidad.
- Los gráficos del EDA se renderizan **en el cliente** (no PNG server-side),
  anticipando la migración de la Fase 4.

**Nota técnica.** `recharts` se fija a la línea **2.x**: la 3.x introduce
react-redux y, con React 18 + Vite dev, provoca "Invalid hook call / more than one
copy of React". `vite.config.ts` deduplica React.

## Pendiente para fases siguientes
- **Fase 4 (UX/optimización)**: mover el render de los gráficos del análisis
  (barras/treemap) a Recharts en el frontend y colapsar los endpoints de
  re-render server-side; wizard EDA→Config→Sesgos→Equidad; code-split del bundle;
  `min_group_size` configurable en la UI.
