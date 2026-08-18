"""
bias_analysis.py — Capa de orquestación y visualización.

El cálculo de métricas se delega en `metrics_core` (pandas/numpy, sin Aequitas).
Este módulo añade solo las visualizaciones server-side (gráficos de distribución
y de disparidad/valores absolutos con matplotlib). En la Fase 4 estos gráficos
se migrarán al frontend (Recharts) y este módulo podrá adelgazarse aún más.
"""
import base64
from io import BytesIO
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import metrics_core
from .metrics_core import (  # noqa: F401  (reexport para compatibilidad)
    recalculate_fairness,
    run_full_analysis as _run_metrics,
)
from .plots import render_group_metric_plot, render_disparity_treemap


def plot_to_base64(fig) -> str:
    """Serializa una figura de Matplotlib a un data URI PNG base64."""
    if fig is None:
        return ""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    buf.seek(0)
    img_bytes = buf.getvalue()
    plt.close(fig)
    plt.close("all")
    if not img_bytes:
        return ""
    return "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")


def generate_distribution_plots(
    df: pd.DataFrame, protected_attributes: List[str], score_col: str, label_col: str
) -> Dict[str, Dict[str, str]]:
    """Gráficos de distribución de predicciones y valores reales por subgrupo."""
    plots: Dict[str, Dict[str, str]] = {}
    # Paleta dimensionada al número de categorías (soporta binario y multiclase).
    n_score = max(2, int(df[score_col].nunique()))
    n_label = max(2, int(df[label_col].nunique()))
    palette_score = sns.diverging_palette(225, 35, n=n_score)
    palette_label = sns.diverging_palette(225, 35, n=n_label)
    for attr in protected_attributes:
        plots[attr] = {}
        fig1 = plt.figure()
        sns.countplot(x=attr, hue=score_col, data=df, palette=palette_score)
        plt.title(f"Distribución de Predicciones por {attr}")
        plt.xticks(rotation=30, ha="right")
        plots[attr]["score_plot"] = plot_to_base64(fig1)

        fig2 = plt.figure()
        sns.countplot(x=attr, hue=label_col, data=df, palette=palette_label)
        plt.title(f"Distribución de Valores Reales por {attr}")
        plt.xticks(rotation=30, ha="right")
        plots[attr]["label_plot"] = plot_to_base64(fig2)
    return plots


def render_absolute_plot(group_metrics_df, metric: str, attribute: str) -> str:
    """Barras de una métrica absoluta por subgrupo, estilo Aequitas."""
    df = pd.DataFrame(group_metrics_df) if not isinstance(group_metrics_df, pd.DataFrame) else group_metrics_df
    return render_group_metric_plot(df, metric, attribute)


def render_disparity_plot(bias_df, metrics: List[str], attribute: str) -> str:
    """Treemap de disparidad por atributo, estilo Aequitas. Usa la 1ª métrica."""
    df = pd.DataFrame(bias_df) if not isinstance(bias_df, pd.DataFrame) else bias_df
    metric = metrics[0] if isinstance(metrics, (list, tuple)) and metrics else metrics
    return render_disparity_treemap(df, metric, attribute)


def run_full_analysis(
    df: pd.DataFrame,
    protected_attributes: List[str],
    score_col: str,
    label_col: str,
    ref_method: str,
    ref_groups: Dict,
    fairness_threshold: float,
    performance_metric: str = "fpr",
    min_group_size: int = 50,
) -> Dict:
    """Orquesta el cálculo (metrics_core) y añade las visualizaciones."""
    results = _run_metrics(
        df=df,
        protected_attributes=protected_attributes,
        score_col=score_col,
        label_col=label_col,
        ref_method=ref_method,
        ref_groups=ref_groups,
        fairness_threshold=fairness_threshold,
        min_group_size=min_group_size,
        performance_metric=performance_metric,
    )

    results["distribution_plots"] = generate_distribution_plots(
        df, protected_attributes, score_col, label_col
    )

    if results["metadata"]["task_type"] == "multiclass":
        # Un gráfico de disparidad inicial por clase (forma binaria por clase).
        for cls, entry in results["by_class"].items():
            entry["plots"] = {
                "disparity_summary": render_disparity_plot(
                    entry["tables"]["bias_metrics"], ["fpr_disparity"], "all"
                )
            }
        results["plots"] = {}
    else:
        results["plots"] = {
            "disparity_summary": render_disparity_plot(
                results["tables"]["bias_metrics"], ["fpr_disparity"], "all"
            )
        }
    return results
