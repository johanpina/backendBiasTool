"""
eda.py — Análisis Exploratorio de Datos (EDA) orientado a sesgos.

Perfila un DataFrame y devuelve JSON (sin gráficos; el frontend los dibuja) para
evidenciar posibles **desbalances** y **sesgos** antes del análisis de equidad:

  - Perfil por columna: tipo, % de nulos, cardinalidad, top-valores, stats.
  - Matriz de asociación (Cramér's V) entre variables categóricas → detecta
    **proxies** (p. ej. "comuna" fuertemente asociada a "nivel socioeconómico").
  - Alertas de calidad: nulos altos, alta cardinalidad, grupos pequeños,
    categorías dominantes (desbalance), columnas constantes y posibles proxies.
"""
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Columnas "categórico-like" para asociación/distribución: pocas categorías.
MAX_CATEGORIES_FOR_ASSOC = 50
# Top-k categorías a devolver por columna (el resto se agrega en "Otros").
TOP_K = 12
# Categorías máximas por eje en las tablas de contingencia (bivariado).
MAX_CATEGORIES_FOR_CROSSTAB = 15
# Máximo de bins en histogramas de variables numéricas.
MAX_HIST_BINS = 20


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _column_dtype(series: pd.Series, n_unique: int) -> str:
    """Clasifica la columna en 'binary' | 'numeric' | 'categorical'."""
    if n_unique <= 2:
        return "binary"
    if _is_numeric(series) and n_unique > 10:
        return "numeric"
    return "categorical"


def _top_values(series: pd.Series, total: int) -> List[Dict]:
    counts = series.value_counts(dropna=True)
    top = counts.head(TOP_K)
    out = [
        {"value": str(idx), "count": int(cnt), "pct": round(100 * cnt / total, 2)}
        for idx, cnt in top.items()
    ]
    remaining = int(counts.iloc[TOP_K:].sum()) if len(counts) > TOP_K else 0
    if remaining:
        out.append({"value": "Otros", "count": remaining,
                    "pct": round(100 * remaining / total, 2), "is_aggregate": True})
    return out


def _normalized_entropy(counts: np.ndarray) -> float:
    """Entropía de Shannon normalizada [0, 1]. 1 = perfectamente balanceada,
    cercano a 0 = una categoría concentra casi todo."""
    total = counts.sum()
    if total <= 0 or len(counts) <= 1:
        return 1.0
    p = counts / total
    p = p[p > 0]
    h = -(p * np.log(p)).sum()
    hmax = math.log(len(counts))
    return float(h / hmax) if hmax > 0 else 1.0


def _balance_label(evenness: float) -> str:
    if evenness >= 0.85:
        return "equilibrada"
    if evenness >= 0.6:
        return "moderada"
    return "desbalanceada"


def _histogram(series: pd.Series) -> List[Dict]:
    """Histograma de una variable numérica: lista de bins {x0, x1, count}."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return []
    n_bins = int(min(MAX_HIST_BINS, max(5, math.ceil(math.sqrt(len(values))))))
    counts, edges = np.histogram(values, bins=n_bins)
    return [
        {"x0": round(float(edges[i]), 4), "x1": round(float(edges[i + 1]), 4),
         "count": int(counts[i])}
        for i in range(len(counts))
    ]


def _role_hint(name: str, dtype: str, n_unique: int, n_rows: int, is_numeric: bool) -> str:
    """Sugerencia heurística del rol de la columna (predicción/etiqueta, protegida,
    identificador o feature). Es solo una pista; el usuario decide."""
    lname = str(name).lower()
    if n_rows and n_unique / n_rows > 0.5 and n_unique > 20:
        return "id"
    if any(k in lname for k in ("score", "pred", "prob", "riesgo", "resultado")):
        return "outcome"
    if any(k in lname for k in ("label", "target", "real", "clase", "y_true", "outcome")):
        return "outcome"
    # Una binaria numérica {0,1} suele ser predicción/etiqueta; una binaria de
    # texto (p. ej. sexo Male/Female) suele ser una variable protegida.
    if dtype == "binary":
        return "outcome" if is_numeric else "protected"
    if dtype == "categorical":
        return "protected"
    return "feature"


def _capped_categories(series: pd.Series, cap: int = MAX_CATEGORIES_FOR_CROSSTAB) -> List[str]:
    """Top-`cap` categorías por frecuencia (como str)."""
    return [str(v) for v in series.value_counts(dropna=True).head(cap).index]


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V corregido (Bergsma) entre dos series categóricas. Rango [0, 1]."""
    confusion = pd.crosstab(x, y)
    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.to_numpy().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return 0.0
    return float(min(1.0, math.sqrt(phi2corr / denom)))


def run_eda(df: pd.DataFrame, min_group_size: int = 50) -> Dict:
    """Genera el perfil exploratorio del DataFrame."""
    n_rows = int(len(df))
    n_cols = int(df.shape[1])
    total_cells = max(1, n_rows * n_cols)
    total_missing = int(df.isna().sum().sum())

    columns: List[Dict] = []
    assoc_cols: List[str] = []
    alerts: List[Dict] = []

    for name in df.columns:
        series = df[name]
        non_null = series.dropna()
        missing = int(series.isna().sum())
        n_unique = int(non_null.nunique())
        dtype = _column_dtype(series, n_unique)

        col: Dict = {
            "name": str(name),
            "dtype": dtype,
            "missing": missing,
            "missing_pct": round(100 * missing / n_rows, 2) if n_rows else 0.0,
            "unique": n_unique,
            "top_values": _top_values(non_null, n_rows) if n_rows else [],
            "stats": None,
        }
        if dtype == "numeric" and len(non_null):
            col["stats"] = {
                "min": float(non_null.min()), "max": float(non_null.max()),
                "mean": float(non_null.mean()), "std": float(non_null.std(ddof=0)),
                "median": float(non_null.median()),
            }
            col["histogram"] = _histogram(non_null)

        # ¿Apta para la matriz de asociación / como variable categórica?
        categorical_like = 2 <= n_unique <= MAX_CATEGORIES_FOR_ASSOC
        col["categorical_like"] = categorical_like
        col["role_hint"] = _role_hint(name, dtype, n_unique, n_rows, _is_numeric(series))

        # Índice de balance (evenness) para variables categóricas.
        if categorical_like:
            vc = non_null.value_counts(dropna=True).to_numpy()
            evenness = _normalized_entropy(vc)
            col["balance"] = {"evenness": round(evenness, 3), "label": _balance_label(evenness)}
        columns.append(col)
        if categorical_like:
            assoc_cols.append(str(name))

        # --- Alertas por columna ---
        if col["missing_pct"] >= 20:
            alerts.append({"level": "critical", "type": "missing", "columns": [str(name)],
                           "message": f"'{name}' tiene {col['missing_pct']:.0f}% de valores faltantes."})
        elif col["missing_pct"] >= 5:
            alerts.append({"level": "warning", "type": "missing", "columns": [str(name)],
                           "message": f"'{name}' tiene {col['missing_pct']:.0f}% de valores faltantes."})

        if n_unique <= 1:
            alerts.append({"level": "warning", "type": "constant", "columns": [str(name)],
                           "message": f"'{name}' es constante (un solo valor); no aporta al análisis."})
        elif n_rows and n_unique / n_rows > 0.5 and n_unique > 20:
            alerts.append({"level": "info", "type": "high_cardinality", "columns": [str(name)],
                           "message": f"'{name}' tiene cardinalidad muy alta ({n_unique} valores); "
                                      f"probablemente sea un identificador y no una variable protegida."})

        if categorical_like and col["top_values"]:
            top = col["top_values"][0]
            if top["pct"] >= 90:
                alerts.append({"level": "warning", "type": "imbalance", "columns": [str(name)],
                               "message": f"'{name}' está muy desbalanceada: '{top['value']}' concentra "
                                          f"el {top['pct']:.0f}% de los casos."})
            small = [tv for tv in col["top_values"]
                     if not tv.get("is_aggregate") and tv["count"] < min_group_size]
            if small:
                grupos = ", ".join(f"{tv['value']} (n={tv['count']})" for tv in small[:5])
                alerts.append({"level": "warning", "type": "small_group", "columns": [str(name)],
                               "message": f"'{name}' tiene subgrupos con muestra pequeña (<{min_group_size}): "
                                          f"{grupos}. Sus métricas serán poco fiables."})

    # --- Matriz de asociación (Cramér's V) entre columnas categórico-like ---
    matrix: List[List[float]] = []
    if len(assoc_cols) >= 2:
        cat_df = df[assoc_cols].astype("object")
        for a in assoc_cols:
            row = []
            for b in assoc_cols:
                if a == b:
                    row.append(1.0)
                else:
                    mask = cat_df[a].notna() & cat_df[b].notna()
                    v = cramers_v(cat_df.loc[mask, a], cat_df.loc[mask, b]) if mask.any() else 0.0
                    row.append(round(v, 3))
            matrix.append(row)

        # Alertas de posible proxy (asociación fuerte entre dos variables).
        for i in range(len(assoc_cols)):
            for j in range(i + 1, len(assoc_cols)):
                v = matrix[i][j]
                if v >= 0.5:
                    level = "warning" if v >= 0.7 else "info"
                    alerts.append({"level": level, "type": "proxy",
                                   "columns": [assoc_cols[i], assoc_cols[j]],
                                   "message": f"'{assoc_cols[i]}' y '{assoc_cols[j]}' están fuertemente "
                                              f"asociadas (V={v:.2f}); una podría ser proxy de la otra."})

    # --- Tablas de contingencia precomputadas (bivariado / sesgo / intersección) ---
    # Solo columnas categórico-like con pocas categorías, para acotar el tamaño.
    crosstab_cols = [c["name"] for c in columns
                     if c["categorical_like"] and c["unique"] <= MAX_CATEGORIES_FOR_CROSSTAB]
    crosstabs: Dict[str, Dict] = {}
    cats_map = {c: _capped_categories(df[c].dropna()) for c in crosstab_cols}
    for a in crosstab_cols:
        for b in crosstab_cols:
            if a == b:
                continue
            xa, yb = cats_map[a], cats_map[b]
            sub = df[[a, b]].dropna().astype("object")
            sub = sub[sub[a].astype(str).isin(xa) & sub[b].astype(str).isin(yb)]
            ct = pd.crosstab(sub[a].astype(str), sub[b].astype(str))
            ct = ct.reindex(index=xa, columns=yb, fill_value=0)
            crosstabs[f"{a}|||{b}"] = {
                "x_values": xa,
                "y_values": yb,
                "counts": ct.to_numpy().astype(int).tolist(),
            }

    # Orden de alertas: critical > warning > info.
    order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: order.get(a["level"], 3))

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_cells_pct": round(100 * total_missing / total_cells, 2),
        "columns": columns,
        "associations": {"columns": assoc_cols, "matrix": matrix},
        "crosstabs": crosstabs,
        "crosstab_columns": crosstab_cols,
        "alerts": alerts,
        "min_group_size": min_group_size,
    }
