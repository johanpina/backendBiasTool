"""
plots.py — Visualizaciones estilo Aequitas con matplotlib puro.

Replica el aspecto de los gráficos originales de Aequitas sin depender de la
librería:
  - `render_group_metric_plot`: barras horizontales de una métrica absoluta,
    agrupadas por atributo, coloreadas por tamaño de grupo y etiquetadas con
    "VALOR (Num: N), X.XX"  (equivalente a `Plot.plot_group_metric`).
  - `render_disparity_treemap`: cuadrícula de treemaps de disparidad, un mapa por
    atributo, con el grupo de referencia marcado "(Ref)" y coloreado por una
    escala divergente centrada en 1  (equivalente a `Plot.plot_disparity_all`).
"""
import base64
import math
from io import BytesIO
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

# Etiquetas legibles de las métricas.
METRIC_LABELS = {
    "accuracy": "Exactitud", "tpr": "TPR", "tnr": "TNR", "for": "FOR",
    "fdr": "FDR", "fpr": "FPR", "fnr": "FNR", "npv": "NPV",
    "precision": "Precisión", "ppr": "PPR", "pprev": "Prev. Predicha", "prev": "Prevalencia",
}
METRIC_FULL = {
    "fpr": "Tasa de Falsos Positivos", "fnr": "Tasa de Falsos Negativos",
    "for": "Tasa de Falsas Omisiones", "fdr": "Tasa de Falsos Descubrimientos",
    "tpr": "Tasa de Verdaderos Positivos", "tnr": "Tasa de Verdaderos Negativos",
    "precision": "Precisión", "npv": "Valor Predictivo Negativo",
    "ppr": "Proporción de Predichos Positivos", "pprev": "Prevalencia Predicha",
    "prev": "Prevalencia", "accuracy": "Exactitud",
}

# Colormap divergente para disparidad: azul (bajo) -> blanco (~1) -> café (alto),
# igual que el estilo de Aequitas.
_DISPARITY_CMAP = LinearSegmentedColormap.from_list(
    "aequitas_disparity", ["#5b8fb0", "#f7f7f7", "#a9763f"]
)
# Colormap secuencial para el tamaño de grupo en el gráfico de barras (tan->café).
_SIZE_CMAP = LinearSegmentedColormap.from_list(
    "group_size", ["#f2d9b1", "#c98a3c", "#5a3d21"]
)


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    data = buf.getvalue()
    plt.close(fig)
    plt.close("all")
    if not data:
        return ""
    return "data:image/png;base64," + base64.b64encode(data).decode("utf-8")


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.upper())


# ---------------------------------------------------------------------------
# Gráfico de métricas absolutas (barras agrupadas por atributo)
# ---------------------------------------------------------------------------
def render_group_metric_plot(df: pd.DataFrame, metric: str, attribute: str) -> str:
    """Barras horizontales de `metric` por subgrupo, agrupadas por atributo."""
    if df is None or metric not in df.columns:
        return ""
    data = df.copy()
    if attribute and attribute not in ("all", "todas"):
        data = data[data["attribute_name"] == attribute]
    if data.empty:
        return ""

    has_size = "group_size" in data.columns
    if has_size:
        smin, smax = data["group_size"].min(), data["group_size"].max()
        size_norm = Normalize(vmin=smin, vmax=smax if smax > smin else smin + 1)

    # Construir posiciones Y: grupos separados por atributo (con un hueco entre atributos).
    attrs = list(dict.fromkeys(data["attribute_name"]))
    y_positions, bar_meta, attr_ticks = [], [], []
    y = 0.0
    for attr in attrs:
        sub = data[data["attribute_name"] == attr]
        start = y
        for _, row in sub.iterrows():
            y_positions.append(y)
            bar_meta.append(row)
            y += 1.0
        attr_ticks.append((attr, (start + y - 1.0) / 2.0))
        y += 1.0  # hueco entre atributos

    height = max(2.5, 0.5 * len(bar_meta) + 0.4 * len(attrs))
    fig, ax = plt.subplots(figsize=(9, height))

    for pos, row in zip(y_positions, bar_meta):
        val = float(row[metric]) if pd.notna(row[metric]) else 0.0
        color = _SIZE_CMAP(size_norm(row["group_size"])) if has_size else "#c98a3c"
        ax.barh(pos, val, height=0.8, color=color, edgecolor="white")
        size_txt = f" (Num: {int(row['group_size']):,})" if has_size else ""
        ax.text(val + 0.015, pos, f"{row['attribute_value']}{size_txt}, {val:.2f}",
                va="center", ha="left", fontsize=9, color="#1f2937")

    # Etiquetas de atributo a la izquierda.
    ax.set_yticks([t[1] for t in attr_ticks])
    ax.set_yticklabels([t[0] for t in attr_ticks], fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Magnitud Absoluta de la Métrica", fontsize=10)
    ax.set_title(f"{_metric_label(metric)} (Modelo 0)", fontsize=13, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Treemap de disparidad (una cuadrícula de mapas, uno por atributo)
# ---------------------------------------------------------------------------
def _squarify(sizes: List[float], x: float, y: float, dx: float, dy: float):
    """Layout de treemap 'squarified'. Devuelve rects [(x, y, dx, dy), ...] en el
    mismo orden que `sizes` (que debe venir ordenado de mayor a menor)."""
    sizes = list(sizes)
    total = sum(sizes)
    if total <= 0:
        return [(x, y, 0, 0) for _ in sizes]
    scaled = [s * dx * dy / total for s in sizes]
    return _squarify_recurse(scaled, [], x, y, dx, dy, [])


def _worst_ratio(row, length, total):
    if length == 0 or not row:
        return math.inf
    side = total / length
    rmax, rmin = max(row), min(row)
    return max((length * length * rmax) / (total * total),
               (total * total * rmin) / (length * length * rmax)) if rmax else math.inf


def _squarify_recurse(scaled, current, x, y, dx, dy, out):
    if not scaled:
        _layout_row(current, x, y, dx, dy, out)
        return out
    length = min(dx, dy)
    total = sum(current)
    nxt = scaled[0]
    if not current or _worst_ratio(current + [nxt], length, total + nxt) <= _worst_ratio(current, length, total):
        _squarify_recurse(scaled[1:], current + [nxt], x, y, dx, dy, out)
    else:
        x, y, dx, dy = _layout_row(current, x, y, dx, dy, out)
        _squarify_recurse(scaled, [], x, y, dx, dy, out)
    return out


def _layout_row(row, x, y, dx, dy, out):
    total = sum(row)
    if total <= 0:
        return x, y, dx, dy
    if dx >= dy:  # apilar verticalmente en una columna de ancho w
        w = total / dy
        cy = y
        for s in row:
            h = s / w
            out.append((x, cy, w, h))
            cy += h
        return x + w, y, dx - w, dy
    else:  # apilar horizontalmente en una fila de alto h
        h = total / dx
        cx = x
        for s in row:
            w = s / h
            out.append((cx, y, w, h))
            cx += w
        return x, y + h, dx, dy - h


def render_disparity_treemap(df: pd.DataFrame, metric: str, attribute: str) -> str:
    """Cuadrícula de treemaps de disparidad, uno por atributo protegido."""
    if isinstance(metric, (list, tuple)):
        metric = metric[0] if metric else "fpr_disparity"
    if df is None or metric not in df.columns:
        return ""
    base = metric.replace("_disparity", "")
    ref_col = f"{base}_ref_group_value"
    data = df.copy()
    if attribute and attribute not in ("all", "todas"):
        data = data[data["attribute_name"] == attribute]
    if data.empty:
        return ""

    attrs = list(dict.fromkeys(data["attribute_name"]))
    ncols = min(3, len(attrs))
    nrows = math.ceil(len(attrs) / ncols)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), squeeze=False)
    unlabeled = []

    for i, attr in enumerate(attrs):
        ax = axes[i // ncols][i % ncols]
        sub = data[data["attribute_name"] == attr].copy()
        if "group_size" in sub.columns:
            sub = sub.sort_values("group_size", ascending=False)
        sizes = sub["group_size"].tolist() if "group_size" in sub.columns else [1] * len(sub)
        rects = _squarify(sizes, 0, 0, 100, 100)

        ref_val = str(sub[ref_col].iloc[0]) if ref_col in sub.columns and len(sub) else None
        for (rx, ry, rw, rh), (_, row) in zip(rects, sub.iterrows()):
            disp = row[metric]
            color = _DISPARITY_CMAP(norm(disp)) if pd.notna(disp) else "#dddddd"
            ax.add_patch(plt.Rectangle((rx, ry), rw, rh, facecolor=color,
                                       edgecolor="white", linewidth=2))
            is_ref = ref_val is not None and str(row["attribute_value"]) == ref_val
            label = f"{row['attribute_value']}\n(Ref)" if is_ref else (
                f"{row['attribute_value']}\n{disp:.2f}" if pd.notna(disp) else f"{row['attribute_value']}")
            # Etiquetar solo si hay espacio horizontal suficiente (evita solapes en
            # rectángulos delgados). La referencia se etiqueta con umbral más laxo.
            if (is_ref and rw >= 7 and rh >= 6) or (rw >= 12 and rh >= 8):
                ax.text(rx + rw / 2, ry + rh / 2, label, ha="center", va="center",
                        fontsize=9, color="#111827")
            elif not is_ref:
                unlabeled.append(f"{attr}: {row['attribute_value']}, {disp:.2f}" if pd.notna(disp) else f"{attr}: {row['attribute_value']}")

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{_metric_label(base)} DISPARITY: {str(attr).upper()}", fontsize=10)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=_DISPARITY_CMAP),
                     ax=ax, fraction=0.046, pad=0.04)

    # Ocultar ejes sobrantes.
    for j in range(len(attrs), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Disparidad de {METRIC_FULL.get(base, base.upper())} A TRAVÉS DE ATRIBUTOS",
                 fontsize=12, y=1.0)
    if unlabeled:
        fig.text(0.01, -0.02, "No etiquetado arriba:\n" + "; ".join(unlabeled),
                 fontsize=8, ha="left", va="top", color="#374151")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _fig_to_base64(fig)
