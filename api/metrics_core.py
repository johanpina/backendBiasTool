"""
metrics_core.py — Motor ligero de métricas de sesgo y equidad.

Reemplaza a la librería Aequitas usando solo pandas/numpy/scipy, preservando
EXACTAMENTE el contrato de columnas que consume el frontend. Replica la tríada
de Aequitas:

  1. get_crosstabs()  -> Group():   conteos y métricas absolutas por subgrupo.
  2. get_disparity()  -> Bias():    disparidades relativas a un grupo de referencia.
  3. _apply_parity()  -> Fairness(): conclusión Fair/Unfair según umbral tau.

Todas las tasas se derivan de la matriz de confusión 2x2 por subgrupo. Las
divisiones por cero producen NaN (igual que Aequitas), no 0 ni inf, salvo el
caso numerador>0 / denominador=0 que produce inf (relevante en disparidades).

Contrato de métricas absolutas (list_absolute_metrics de Aequitas, 12):
    accuracy, tpr, tnr, for, fdr, fpr, fnr, npv, precision, ppr, pprev, prev

Contrato de disparidades (10, Aequitas NO genera accuracy_disparity):
    ppr, pprev, precision, fdr, for, fpr, fnr, tpr, tnr, npv
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Las 12 métricas absolutas por subgrupo, en el orden de Aequitas.
ABSOLUTE_METRICS: List[str] = [
    "accuracy", "tpr", "tnr", "for", "fdr", "fpr", "fnr",
    "npv", "precision", "ppr", "pprev", "prev",
]

# Las 10 métricas para las que se calcula disparidad (Aequitas no incluye accuracy).
DISPARITY_METRICS: List[str] = [
    "ppr", "pprev", "precision", "fdr", "for", "fpr", "fnr", "tpr", "tnr", "npv",
]

# Métricas de error que definen la CONCLUSIÓN de equidad por subgrupo. Son las
# que la herramienta prioriza (FPR, FNR, FOR, FDR): tasas independientes del
# tamaño del grupo, por lo que responden al umbral de tolerancia. En cambio la
# Paridad Estadística (ppr_disparity) depende del tamaño relativo del grupo y
# casi siempre da "Unfair", por eso NO se usa para la conclusión general.
FAIRNESS_CONCLUSION_METRICS: List[str] = ["fpr", "fnr", "for", "fdr"]

# Tamaño mínimo de subgrupo para que su disparidad sea estadísticamente fiable.
# Los subgrupos por debajo de este umbral se marcan "muestra insuficiente" y NO
# determinan la conclusión de equidad del atributo (siguen visibles en las tablas).
DEFAULT_MIN_GROUP_SIZE: int = 50

# Mapa disparidad -> nombre de columna de paridad Fair/Unfair (idéntico a Aequitas).
PARITY_MAPPING: Dict[str, str] = {
    "ppr_disparity": "Statistical Parity",
    "pprev_disparity": "Impact Parity",
    "fdr_disparity": "FDR Parity",
    "for_disparity": "FOR Parity",
    "fpr_disparity": "FPR Parity",
    "fnr_disparity": "FNR Parity",
    "tpr_disparity": "TPR Parity",
    "tnr_disparity": "TNR Parity",
    "npv_disparity": "NPV Parity",
    "precision_disparity": "Precision Parity",
}


def _safe_divide(numerator, denominator):
    """División elemento a elemento que devuelve NaN donde el denominador es 0.

    Replica el comportamiento de Aequitas para métricas absolutas (0/0 -> NaN).
    """
    numerator = np.asarray(numerator, dtype="float64")
    denominator = np.asarray(denominator, dtype="float64")
    out = np.full(numerator.shape, np.nan, dtype="float64")
    return np.divide(numerator, denominator, out=out, where=denominator != 0)


# ---------------------------------------------------------------------------
# 1. Group(): conteos + métricas absolutas
# ---------------------------------------------------------------------------
def get_crosstabs(
    df: pd.DataFrame,
    attribute_names: List[str],
    score_col: str = "score",
    label_col: str = "label_value",
) -> Tuple[pd.DataFrame, List[str]]:
    """Calcula conteos y métricas absolutas por subgrupo.

    Equivalente a ``Group().get_crosstabs``. Devuelve ``(xtab, absolute_metrics)``
    donde ``xtab`` tiene una fila por ``(attribute_name, attribute_value)``.

    Espera ``score`` y ``label_value`` binarios en {0, 1}.
    """
    scores = pd.to_numeric(df[score_col], errors="coerce").astype("float64")
    labels = pd.to_numeric(df[label_col], errors="coerce").astype("float64")

    _validate_binary(scores, score_col)
    _validate_binary(labels, label_col)

    total_entities = len(df)
    rows = []

    for attr in attribute_names:
        values = df[attr]
        # Excluir filas con atributo protegido nulo para ese atributo.
        mask_notna = values.notna()
        sub_scores = scores[mask_notna]
        sub_labels = labels[mask_notna]
        sub_values = values[mask_notna]

        tp = (sub_scores == 1) & (sub_labels == 1)
        fp = (sub_scores == 1) & (sub_labels == 0)
        tn = (sub_scores == 0) & (sub_labels == 0)
        fn = (sub_scores == 0) & (sub_labels == 1)

        grp = pd.DataFrame({
            "attribute_value": sub_values.values,
            "tp": tp.values.astype(int),
            "fp": fp.values.astype(int),
            "tn": tn.values.astype(int),
            "fn": fn.values.astype(int),
        }).groupby("attribute_value", sort=True).sum().reset_index()

        grp.insert(0, "attribute_name", attr)
        rows.append(grp)

    xtab = pd.concat(rows, ignore_index=True)

    # Conteos derivados.
    xtab["pp"] = xtab["tp"] + xtab["fp"]              # predichos positivos
    xtab["pn"] = xtab["tn"] + xtab["fn"]              # predichos negativos
    xtab["group_label_pos"] = xtab["tp"] + xtab["fn"]  # positivos reales
    xtab["group_label_neg"] = xtab["tn"] + xtab["fp"]  # negativos reales
    xtab["group_size"] = xtab["tp"] + xtab["fp"] + xtab["tn"] + xtab["fn"]
    xtab["total_entities"] = total_entities

    # Columnas de compatibilidad con el formato de Aequitas.
    xtab.insert(0, "model_id", 0)
    xtab.insert(1, "score_threshold", "binary 0/1")
    xtab.insert(2, "k", xtab["pp"])

    xtab = _absolute_metrics(xtab)

    # Orden de columnas: conteos primero, luego métricas absolutas.
    count_cols = [
        "model_id", "score_threshold", "k", "attribute_name", "attribute_value",
        "pp", "pn", "fp", "fn", "tn", "tp",
        "group_label_pos", "group_label_neg", "group_size", "total_entities",
    ]
    xtab = xtab[count_cols + ABSOLUTE_METRICS]
    return xtab, list(ABSOLUTE_METRICS)


def _absolute_metrics(xtab: pd.DataFrame) -> pd.DataFrame:
    """Añade las 12 métricas absolutas a partir de la matriz 2x2 por grupo."""
    tp, fp, tn, fn = xtab["tp"], xtab["fp"], xtab["tn"], xtab["fn"]
    pp = xtab["pp"]
    group_size = xtab["group_size"]
    group_label_pos = xtab["group_label_pos"]

    xtab["tpr"] = _safe_divide(tp, tp + fn)
    xtab["tnr"] = _safe_divide(tn, tn + fp)
    xtab["fpr"] = _safe_divide(fp, fp + tn)
    xtab["fnr"] = _safe_divide(fn, fn + tp)
    xtab["precision"] = _safe_divide(tp, tp + fp)
    xtab["fdr"] = _safe_divide(fp, tp + fp)
    xtab["for"] = _safe_divide(fn, tn + fn)
    xtab["npv"] = _safe_divide(tn, tn + fn)
    xtab["pprev"] = _safe_divide(pp, group_size)
    xtab["prev"] = _safe_divide(group_label_pos, group_size)
    xtab["accuracy"] = _safe_divide(tp + tn, group_size)

    # PPR: proporción de predichos positivos del grupo respecto al total de
    # predichos positivos DENTRO del mismo atributo (comportamiento de Aequitas).
    pp_total_by_attr = xtab.groupby("attribute_name")["pp"].transform("sum")
    xtab["ppr"] = _safe_divide(pp, pp_total_by_attr)
    return xtab


def _validate_binary(series: pd.Series, name: str) -> None:
    unique = set(pd.unique(series.dropna()))
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(
            f"La columna '{name}' debe ser binaria (valores 0/1). "
            f"Valores encontrados: {sorted(unique)[:10]}"
        )


# ---------------------------------------------------------------------------
# 2. Bias(): selección de grupo de referencia + disparidades
# ---------------------------------------------------------------------------
def _reference_rows(
    xtab: pd.DataFrame,
    ref_method: str,
    ref_groups: Optional[Dict[str, str]],
    performance_metric: str,
) -> Dict[str, str]:
    """Devuelve ``{attribute_name: attribute_value_referencia}`` por atributo.

    Estrategias:
      - "majority": grupo con mayor group_size.
      - "minority": grupo con menor group_size.
      - "custom":   valor indicado en ref_groups (fallback a majority si falta).
      - "best_performance": grupo con MENOR valor de ``performance_metric``
        (replica literal de get_disparity_min_metric de Aequitas).

    Desempate estable por orden alfabético del attribute_value.
    """
    ref_map: Dict[str, str] = {}
    ref_groups = ref_groups or {}

    for attr, grp in xtab.groupby("attribute_name", sort=True):
        grp = grp.sort_values("attribute_value")  # desempate determinista

        if ref_method == "custom" and attr in ref_groups:
            candidate = ref_groups[attr]
            valid = set(grp["attribute_value"].astype(str))
            if str(candidate) not in valid:
                raise ValueError(
                    f"El grupo de referencia '{candidate}' no existe para el "
                    f"atributo '{attr}'. Valores válidos: {sorted(valid)}"
                )
            ref_map[attr] = candidate
        elif ref_method == "minority":
            ref_map[attr] = grp.loc[grp["group_size"].idxmin(), "attribute_value"]
        elif ref_method == "best_performance":
            metric_vals = grp[performance_metric]
            if metric_vals.notna().any():
                ref_map[attr] = grp.loc[metric_vals.idxmin(), "attribute_value"]
            else:  # todas NaN -> fallback a majority
                ref_map[attr] = grp.loc[grp["group_size"].idxmax(), "attribute_value"]
        else:  # "majority" (default)
            ref_map[attr] = grp.loc[grp["group_size"].idxmax(), "attribute_value"]

    return ref_map


# Contingencia por métrica: (positivos, negativos) para el chi-cuadrado 2x2.
_CONTINGENCY_COLS = {
    "ppr": ("pp", "pn"),
    "pprev": ("pp", "pn"),
    "precision": ("tp", "fp"),
    "fdr": ("fp", "tp"),
    "for": ("fn", "tn"),
    "npv": ("tn", "fn"),
    "fpr": ("fp", "tn"),
    "tnr": ("tn", "fp"),
    "fnr": ("fn", "tp"),
    "tpr": ("tp", "fn"),
}


def get_disparity(
    xtab: pd.DataFrame,
    ref_map: Dict[str, str],
    disparity_metrics: Optional[List[str]] = None,
    alpha: float = 0.05,
    mask_significance: bool = True,
) -> pd.DataFrame:
    """Disparidades relativas a un ÚNICO grupo de referencia por atributo.

    Estrategias majority/minority/custom. Para cada métrica ``m``:
    ``{m}_disparity = metric_grupo / metric_referencia``.
    """
    ref_value_for = lambda attr, m: ref_map.get(attr)
    return _compute_disparities(xtab, ref_value_for, disparity_metrics, mask_significance, alpha)


def get_disparity_min_metric(
    xtab: pd.DataFrame,
    disparity_metrics: Optional[List[str]] = None,
    alpha: float = 0.05,
    mask_significance: bool = True,
) -> pd.DataFrame:
    """Disparidades usando, para CADA métrica, el grupo con el valor MÍNIMO de
    esa métrica como referencia (replica ``Bias.get_disparity_min_metric`` de
    Aequitas). La referencia varía por métrica, no es única por atributo.
    """
    if disparity_metrics is None:
        disparity_metrics = DISPARITY_METRICS

    # Referencia por (atributo, métrica) = grupo con idxmin de la métrica.
    ref_by_attr_metric: Dict = {}
    for attr, grp in xtab.groupby("attribute_name", sort=True):
        grp = grp.sort_values("attribute_value")
        for m in disparity_metrics:
            col = grp[m]
            if col.notna().any():
                ref_val = grp.loc[col.idxmin(), "attribute_value"]
            else:  # métrica toda NaN -> primer grupo (fallback, como Aequitas)
                ref_val = grp["attribute_value"].iloc[0]
            ref_by_attr_metric[(attr, m)] = ref_val

    ref_value_for = lambda attr, m: ref_by_attr_metric.get((attr, m))
    return _compute_disparities(xtab, ref_value_for, disparity_metrics, mask_significance, alpha)


def _compute_disparities(
    xtab: pd.DataFrame,
    ref_value_for,
    disparity_metrics: Optional[List[str]],
    mask_significance: bool,
    alpha: float,
) -> pd.DataFrame:
    """Núcleo de cálculo de disparidades dado un selector de referencia.

    ``ref_value_for(attr, metric)`` devuelve el ``attribute_value`` de referencia
    para ese atributo y métrica (constante entre métricas para las estrategias
    de referencia única; variable para min-metric).
    """
    if disparity_metrics is None:
        disparity_metrics = DISPARITY_METRICS

    bias_df = xtab.copy()
    # Índice rápido: (attribute_name, attribute_value) -> fila.
    indexed = xtab.set_index(["attribute_name", "attribute_value"])

    for m in disparity_metrics:
        disparity = np.full(len(bias_df), np.nan)
        ref_group_val = np.empty(len(bias_df), dtype=object)
        sig = pd.Series([np.nan] * len(bias_df), index=bias_df.index, dtype="object")
        pos_c, neg_c = _CONTINGENCY_COLS.get(m, (None, None))

        for pos, (_, row) in enumerate(bias_df.iterrows()):
            attr = row["attribute_name"]
            ref_val = ref_value_for(attr, m)
            ref_group_val[pos] = str(ref_val) if ref_val is not None else ""
            if ref_val is None or (attr, ref_val) not in indexed.index:
                continue
            ref_row = indexed.loc[(attr, ref_val)]
            disparity[pos] = _safe_divide([row[m]], ref_row[m])[0]

            if mask_significance and pos_c is not None:
                sig.iloc[pos] = _chi2_significant(row, ref_row, pos_c, neg_c, alpha)

        bias_df[f"{m}_disparity"] = disparity
        bias_df[f"{m}_ref_group_value"] = ref_group_val
        if mask_significance:
            bias_df[f"{m}_significance"] = sig

    return bias_df


def _chi2_significant(row, ref_row, pos_c, neg_c, alpha):
    """Test chi-cuadrado 2x2 grupo vs. referencia. NaN si no es válido."""
    table = np.array([
        [float(row[pos_c]), float(row[neg_c])],
        [float(ref_row[pos_c]), float(ref_row[neg_c])],
    ])
    try:
        if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
            return np.nan
        _, p, _, _ = stats.chi2_contingency(table)
        return bool(p < alpha)
    except (ValueError, ZeroDivisionError):
        return np.nan


# ---------------------------------------------------------------------------
# 3. Fairness(): paridades Fair/Unfair
# ---------------------------------------------------------------------------
def _apply_parity(
    bias_df: pd.DataFrame,
    fairness_threshold: float,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Añade columnas de paridad Fair/Unfair y ``fairness_conclusion``.

    Fair si la disparidad es NaN o cae en ``[1/tau, tau]``. Un valor infinito
    (numerador>0, denominador de referencia=0) queda fuera del rango -> Unfair.
    Marca ``insufficient_sample`` para subgrupos con ``group_size < min_group_size``.
    """
    fairness_df = bias_df.copy()
    lower_bound = 1.0 / fairness_threshold
    upper_bound = fairness_threshold

    if "group_size" in fairness_df.columns:
        fairness_df["insufficient_sample"] = fairness_df["group_size"] < min_group_size

    for disparity_metric, parity_metric in PARITY_MAPPING.items():
        if disparity_metric in fairness_df.columns:
            fairness_df[parity_metric] = fairness_df[disparity_metric].apply(
                lambda x: "Fair" if pd.isna(x) or (lower_bound <= x <= upper_bound) else "Unfair"
            )

    # La conclusión general del subgrupo se basa en las métricas de error
    # (FPR/FNR/FOR/FDR): "Unfair" si CUALQUIERA de ellas supera la tolerancia.
    conclusion_parities = [
        PARITY_MAPPING[f"{m}_disparity"]
        for m in FAIRNESS_CONCLUSION_METRICS
        if f"{m}_disparity" in PARITY_MAPPING and PARITY_MAPPING[f"{m}_disparity"] in fairness_df.columns
    ]
    if conclusion_parities:
        fairness_df["fairness_conclusion"] = fairness_df[conclusion_parities].apply(
            lambda row: "Unfair" if (row == "Unfair").any() else "Fair", axis=1
        )
    elif "Statistical Parity" in fairness_df.columns:
        fairness_df["fairness_conclusion"] = fairness_df["Statistical Parity"]
    return fairness_df


def _reliable_rows(group: pd.DataFrame) -> pd.DataFrame:
    """Filtra subgrupos con muestra suficiente; si todos son insuficientes,
    devuelve el grupo completo (fallback para no perder el atributo)."""
    if "insufficient_sample" not in group.columns:
        return group
    reliable = group[~group["insufficient_sample"].astype(bool)]
    return reliable if not reliable.empty else group


def _fairness_summary(fairness_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen por atributo: Unfair si algún subgrupo FIABLE del atributo es Unfair.

    Los subgrupos con muestra insuficiente no determinan la conclusión.
    """
    rows = []
    for attr, group in fairness_df.groupby("attribute_name", sort=True):
        reliable = _reliable_rows(group)
        verdict = "Unfair" if (reliable["fairness_conclusion"] == "Unfair").any() else "Fair"
        rows.append({"attribute_name": attr, "fairness_conclusion": verdict})
    return pd.DataFrame(rows)


def get_group_attribute_fairness(fairness_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las paridades a nivel de atributo (Unfair si algún subgrupo fiable
    lo es), ignorando subgrupos con muestra insuficiente."""
    parity_cols = [c for c in PARITY_MAPPING.values() if c in fairness_df.columns]
    include_conclusion = "fairness_conclusion" in fairness_df.columns
    rows = []
    for attr, group in fairness_df.groupby("attribute_name", sort=True):
        reliable = _reliable_rows(group)
        row = {"attribute_name": attr}
        for col in parity_cols:
            row[col] = "Unfair" if (reliable[col] == "Unfair").any() else "Fair"
        if include_conclusion:
            row["fairness_conclusion"] = (
                "Unfair" if (reliable["fairness_conclusion"] == "Unfair").any() else "Fair"
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detección de tarea y binarización one-vs-rest (multiclase)
# ---------------------------------------------------------------------------
def _detect_task(label_series: pd.Series, score_series: pd.Series):
    """Devuelve ``('binary'|'multiclass', classes)``.

    Binario si hay como máximo 2 clases y son numéricas {0, 1}; en caso contrario
    (más de 2 clases, o 2 clases no {0,1}) se trata como multiclase one-vs-rest.
    """
    vals = pd.concat([label_series, score_series]).dropna()
    classes = sorted(pd.unique(vals), key=lambda x: str(x))
    coerced = pd.to_numeric(vals, errors="coerce")
    all_numeric = bool(coerced.notna().all())
    numeric_vals = set(coerced.dropna().unique())
    if len(classes) <= 2 and all_numeric and numeric_vals <= {0.0, 1.0}:
        return "binary", classes
    return "multiclass", classes


def _binarize_ovr(df_local: pd.DataFrame, positive_class) -> pd.DataFrame:
    """Binariza score/label como 'clase vs. resto' (1 si == positive_class)."""
    out = df_local.copy()
    out["score"] = (df_local["score"].astype(object) == positive_class).astype(int)
    out["label_value"] = (df_local["label_value"].astype(object) == positive_class).astype(int)
    return out


def _run_binary_tables(
    df_local: pd.DataFrame,
    protected_attributes: List[str],
    ref_method: str,
    ref_groups: Dict,
    fairness_threshold: float,
    performance_metric: str,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> Dict[str, pd.DataFrame]:
    """Núcleo de cálculo binario. Devuelve el dict de tablas del contrato."""
    xtab, absolute_metrics = get_crosstabs(df_local, protected_attributes)

    group_counts_df = xtab[[c for c in xtab.columns if c not in absolute_metrics]]
    group_metrics_df = xtab[["attribute_name", "attribute_value"] + absolute_metrics]
    group_metrics_for_plotting = group_metrics_df.merge(
        group_counts_df[["attribute_name", "attribute_value", "group_size"]],
        on=["attribute_name", "attribute_value"],
    )

    if ref_method == "best_performance":
        bias_df = get_disparity_min_metric(xtab, mask_significance=True)
    else:
        ref_map = _reference_rows(xtab, ref_method, ref_groups, performance_metric)
        bias_df = get_disparity(xtab, ref_map, mask_significance=True)

    fairness_df = _apply_parity(bias_df, fairness_threshold, min_group_size)
    return {
        "group_counts": group_counts_df,
        "group_metrics": group_metrics_df,
        "group_metrics_for_plotting": group_metrics_for_plotting,
        "bias_metrics": fairness_df,
        "fairness_summary": _fairness_summary(fairness_df),
        "fairness_by_attribute": get_group_attribute_fairness(fairness_df),
    }


def _fairness_overall(by_class: Dict[str, Dict]) -> pd.DataFrame:
    """Resumen multiclase: Unfair para un atributo si lo es en CUALQUIER clase."""
    frames = [entry["tables"]["fairness_summary"] for entry in by_class.values()]
    allfs = pd.concat(frames, ignore_index=True)
    return (
        allfs.groupby("attribute_name")
        .agg(fairness_conclusion=("fairness_conclusion",
                                  lambda x: "Unfair" if "Unfair" in x.values else "Fair"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# API pública (firma idéntica a la del módulo Aequitas anterior)
# ---------------------------------------------------------------------------
def run_full_analysis(
    df: pd.DataFrame,
    protected_attributes: List[str],
    score_col: str,
    label_col: str,
    ref_method: str,
    ref_groups: Dict,
    fairness_threshold: float,
    performance_metric: str = "fpr",
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> Dict:
    """Ejecuta el análisis completo (binario o multiclase) y devuelve las tablas.

    - **Binario** ({0,1}): estructura idéntica a la de Aequitas (sin cambios para
      el frontend).
    - **Multiclase** (>2 clases): estrategia one-vs-rest. Cada tabla lleva una
      columna ``class`` (concatenada) y además se expone ``by_class`` con las
      tablas por clase (forma binaria) y ``fairness_overall`` (Unfair si lo es en
      alguna clase).
    """
    task_type, classes = _detect_task(df[label_col], df[score_col])
    unique_values = {col: df[col].dropna().unique().tolist() for col in protected_attributes}
    metadata = {
        "protected_attributes": protected_attributes,
        "unique_values": unique_values,
        "fairness_threshold": fairness_threshold,
        "min_group_size": min_group_size,
        "task_type": task_type,
        # Método de selección del grupo de referencia (para mostrarlo en la UI y el PDF).
        "ref_method": ref_method,
        "performance_metric": performance_metric,
    }

    df_local = df[protected_attributes].copy()

    if task_type == "binary":
        df_local["score"] = pd.to_numeric(df[score_col], errors="coerce")
        df_local["label_value"] = pd.to_numeric(df[label_col], errors="coerce")
        tables = _run_binary_tables(df_local, protected_attributes, ref_method,
                                    ref_groups, fairness_threshold, performance_metric,
                                    min_group_size)
        return {"distribution_plots": {}, "plots": {}, "tables": tables, "metadata": metadata}

    # --- Multiclase (one-vs-rest) ---
    df_local["score"] = df[score_col].values
    df_local["label_value"] = df[label_col].values

    table_keys = ["group_counts", "group_metrics", "group_metrics_for_plotting",
                  "bias_metrics", "fairness_summary", "fairness_by_attribute"]
    by_class: Dict[str, Dict] = {}
    concat: Dict[str, list] = {k: [] for k in table_keys}

    for c in classes:
        bin_df = _binarize_ovr(df_local, c)
        t = _run_binary_tables(bin_df, protected_attributes, ref_method,
                               ref_groups, fairness_threshold, performance_metric,
                               min_group_size)
        by_class[str(c)] = {"tables": t, "plots": {}}
        for k, tbl in t.items():
            tagged = tbl.copy()
            tagged.insert(0, "class", str(c))
            concat[k].append(tagged)

    tables = {k: pd.concat(concat[k], ignore_index=True) for k in table_keys}
    metadata["classes"] = [str(c) for c in classes]

    return {
        "distribution_plots": {},
        "plots": {},
        "tables": tables,
        "by_class": by_class,
        "fairness_overall": _fairness_overall(by_class),
        "metadata": metadata,
    }


def recalculate_fairness(
    bias_df: pd.DataFrame,
    fairness_threshold: float,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> Dict:
    """Recalcula solo las paridades con un nuevo umbral (sin recomputar métricas)."""
    fairness_df = _apply_parity(bias_df, fairness_threshold, min_group_size)
    fairness_summary_df = _fairness_summary(fairness_df)
    fairness_by_attribute_df = get_group_attribute_fairness(fairness_df)
    return {
        "fairness_summary": fairness_summary_df,
        "fairness_by_attribute": fairness_by_attribute_df,
    }
