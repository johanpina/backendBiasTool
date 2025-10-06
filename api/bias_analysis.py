
import sys
import types
import pandas as pd
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.plotting import Plot
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

from aequitas.fairness import Fairness

# --- Inicio del Parche para `fairgbm` ---
dummy_module = types.ModuleType('fairgbm')
class DummyFairGBMClassifier: pass
dummy_module.FairGBMClassifier = DummyFairGBMClassifier
sys.modules['fairgbm'] = dummy_module
# --- Fin del Parche ---

# Diccionario de traducciones para los gráficos y tablas
TRANSLATION_MAP = {
    # Métricas
    'accuracy': 'Exactitud',
    'tpr': 'Tasa Verdaderos Positivos (TPR)',
    'tnr': 'Tasa Verdaderos Negativos (TNR)',
    'for': 'Tasa Falsas Omisiones (FOR)',
    'fdr': 'Tasa Falsos Descubrimientos (FDR)',
    'fpr': 'Tasa Falsos Positivos (FPR)',
    'fnr': 'Tasa Falsos Negativos (FNR)',
    'npv': 'Valor Predictivo Negativo (NPV)',
    'precision': 'Precisión',
    'ppr': 'Proporción Predicha Positiva (PPR)',
    'pprev': 'Prevalencia Predicha',
    # Disparidades
    'PPREV_DISPARITY': 'Disparidad de Prevalencia Predicha',
    'FDR_DISPARITY': 'Disparidad de Tasa de Falsos Descubrimientos',
    'PRECISION_DISPARITY': 'Disparidad de Precisión',
    'FOR_DISPARITY': 'Disparidad de Tasa de Falsas Omisiones',
    'FPR_DISPARITY': 'Disparidad de Tasa de Falsos Positivos',
    'FNR_DISPARITY': 'Disparidad de Tasa de Falsos Negativos',
    'TPR_DISPARITY': 'Disparidad de Tasa de Verdaderos Positivos',
    'TNR_DISPARITY': 'Disparidad de Tasa de Verdaderos Negativos',
    'NPV_DISPARITY': 'Disparidad de Valor Predictivo Negativo',
    # Textos generales del gráfico
    'ACROSS ATTRIBUTES': 'A TRAVÉS DE ATRIBUTOS',
    'RAZA': 'RAZA',
    'GENERO': 'GÉNERO',
    'CATEGORIA_EDAD': 'CATEGORÍA DE EDAD',
    'Not labeled above:': 'No etiquetado arriba:',
    'Absolute Metric Magnitude': 'Magnitud Absoluta de la Métrica',
    '(Model 0)': '(Modelo 0)'
}

def translate_plot_text(figure):
    """Recorre los elementos de texto de una figura de Matplotlib y los traduce."""
    # Traducir el título principal si existe
    if figure._suptitle:
        original_title = figure._suptitle.get_text()
        for eng, esp in TRANSLATION_MAP.items():
            original_title = original_title.replace(eng, esp)
        figure.suptitle(original_title)

    for ax in figure.axes:
        # Traducir títulos de subplots
        if ax.get_title():
            original_ax_title = ax.get_title()
            for eng, esp in TRANSLATION_MAP.items():
                original_ax_title = original_ax_title.replace(eng, esp)
            ax.set_title(original_ax_title)
        
        # Traducir etiquetas de ejes
        if ax.get_xlabel():
            original_xlabel = ax.get_xlabel()
            for eng, esp in TRANSLATION_MAP.items():
                original_xlabel = original_xlabel.replace(eng, esp)
            ax.set_xlabel(original_xlabel)
        
        if ax.get_ylabel():
            original_ylabel = ax.get_ylabel()
            for eng, esp in TRANSLATION_MAP.items():
                original_ylabel = original_ylabel.replace(eng, esp)
            ax.set_ylabel(original_ylabel)

        # Traducir otras anotaciones de texto
        for text_obj in ax.texts:
            original_text = text_obj.get_text()
            for eng, esp in TRANSLATION_MAP.items():
                original_text = original_text.replace(eng, esp)
            text_obj.set_text(original_text)
    return figure

def plot_to_base64(plt_object) -> str:
    # Aequitas puede devolver None si no puede generar un gráfico para una combinación.
    # Con esto, manejamos el caso y devolvemos una imagen vacía en lugar de un error.
    if plt_object is None:
        return ""

    buf = BytesIO()
    figure = plt_object.figure if hasattr(plt_object, 'figure') else plt_object
    
    # Doble chequeo por si el objeto figura interno es nulo
    if figure is None:
        return ""

    # Aplicar traducciones antes de guardar
    figure = translate_plot_text(figure)

    figure.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.getvalue()
    if not img_bytes:
        raise ValueError("Error al generar la imagen del gráfico: el buffer de imagen está vacío.")
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    plt.close('all')
    return f"data:image/png;base64,{img_base64}"

def generate_distribution_plots(df: pd.DataFrame, protected_attributes: List[str], score_col: str, label_col: str) -> Dict[str, Dict[str, str]]:
    plots = {}
    palette = sns.diverging_palette(225, 35, n=2)
    for attr in protected_attributes:
        plots[attr] = {}
        fig1 = plt.figure()
        sns.countplot(x=attr, hue=score_col, data=df, palette=palette)
        plt.title(f'Distribución de Predicciones por {attr}')
        plots[attr]['score_plot'] = plot_to_base64(fig1)
        plt.close(fig1)
        fig2 = plt.figure()
        sns.countplot(x=attr, hue=label_col, data=df, palette=palette)
        plt.title(f'Distribución de Valores Reales por {attr}')
        plots[attr]['label_plot'] = plot_to_base64(fig2)
        plt.close(fig2)
    return plots

def render_disparity_plot(bias_df: pd.DataFrame, metrics: List[str], attribute: str) -> str:
    aqp = Plot()
    if 'model_id' in bias_df.columns and bias_df['model_id'].nunique() > 1:
        model_to_plot = bias_df['model_id'].unique()[0]
        bias_df = bias_df[bias_df['model_id'] == model_to_plot]

    if attribute == 'all':
        fig = aqp.plot_disparity_all(bias_df, metrics=metrics, significance_alpha=0.05)
    else:
        fig = aqp.plot_disparity(bias_df, group_metric=metrics[0], attribute_name=attribute, significance_alpha=0.05)
        
    return plot_to_base64(fig)

def render_absolute_plot(group_metrics_df: pd.DataFrame, metric: str, attribute: str) -> str:
    aqp = Plot()
    df_to_plot = group_metrics_df if attribute == 'all' else group_metrics_df[group_metrics_df['attribute_name'] == attribute]
    fig = aqp.plot_group_metric(df_to_plot, metric)
    return plot_to_base64(fig)

def recalculate_fairness(bias_df: pd.DataFrame, fairness_threshold: float):
    """Recalcula solo las métricas de equidad con un nuevo umbral."""
    fairness_df = bias_df.copy()
    parity_mapping = {
        'ppr_disparity': 'Statistical Parity', 'pprev_disparity': 'Impact Parity',
        'fdr_disparity': 'FDR Parity', 'for_disparity': 'FOR Parity',
        'fpr_disparity': 'FPR Parity', 'fnr_disparity': 'FNR Parity',
        'tpr_disparity': 'TPR Parity', 'tnr_disparity': 'TNR Parity',
        'npv_disparity': 'NPV Parity', 'precision_disparity': 'Precision Parity',
        'accuracy_disparity': 'Accuracy Parity'
    }
    lower_bound = 1 / fairness_threshold
    upper_bound = fairness_threshold
    for disparity_metric, parity_metric in parity_mapping.items():
        if disparity_metric in fairness_df.columns:
            fairness_df[parity_metric] = fairness_df[disparity_metric].apply(
                lambda x: 'Fair' if pd.isna(x) or (lower_bound <= x <= upper_bound) else 'Unfair'
            )
    if 'Statistical Parity' in fairness_df.columns:
      fairness_df['fairness_conclusion'] = fairness_df['Statistical Parity']
    
    fairness_summary_df = fairness_df.groupby('attribute_name').agg(
        fairness_conclusion=('fairness_conclusion', lambda x: 'Unfair' if 'Unfair' in x.values else 'Fair')
    ).reset_index()

    f = Fairness()
    fairness_by_attribute_df = f.get_group_attribute_fairness(fairness_df)

    return {
        "fairness_summary": fairness_summary_df,
        "fairness_by_attribute": fairness_by_attribute_df
    }


def run_full_analysis(df: pd.DataFrame, protected_attributes: List[str], score_col: str, label_col: str, ref_method: str, ref_groups: Dict, fairness_threshold: float, performance_metric: str = 'fpr'):
    df_aequitas = df[protected_attributes].copy()
    df_aequitas['score'] = df[score_col]
    df_aequitas['label_value'] = df[label_col]

    dist_plots = generate_distribution_plots(df, protected_attributes, score_col, label_col)

    g = Group()
    xtab, _ = g.get_crosstabs(df_aequitas)
    absolute_metrics = g.list_absolute_metrics(xtab)
    
    group_counts_df = xtab[[c for c in xtab.columns if c not in absolute_metrics]]
    group_metrics_df = xtab[['attribute_name', 'attribute_value'] + absolute_metrics]
    group_metrics_for_plotting = group_metrics_df.merge(group_counts_df[['attribute_name', 'attribute_value', 'group_size']], on=['attribute_name', 'attribute_value'])

    b = Bias()
    # Lógica condicional para seleccionar el método de cálculo de disparidad
    if ref_method == "custom" and ref_groups:
        bias_df = b.get_disparity_predefined_groups(xtab, original_df=df_aequitas, ref_groups_dict=ref_groups)
    elif ref_method == "best_performance":
        bias_df = b.get_disparity_min_metric(xtab, original_df=df_aequitas, label_score_ref=performance_metric)
    else: # Por defecto o si es 'majority'
        bias_df = b.get_disparity_major_group(xtab, original_df=df_aequitas)

    if 'model_id' in bias_df.columns and bias_df['model_id'].nunique() > 1:
        model_to_use = bias_df['model_id'].unique()[0]
        bias_df = bias_df[bias_df['model_id'] == model_to_use]

    fairness_df = bias_df.copy()
    parity_mapping = {
        'ppr_disparity': 'Statistical Parity', 'pprev_disparity': 'Impact Parity',
        'fdr_disparity': 'FDR Parity', 'for_disparity': 'FOR Parity',
        'fpr_disparity': 'FPR Parity', 'fnr_disparity': 'FNR Parity',
        'tpr_disparity': 'TPR Parity', 'tnr_disparity': 'TNR Parity',
        'npv_disparity': 'NPV Parity', 'precision_disparity': 'Precision Parity',
        'accuracy_disparity': 'Accuracy Parity'
    }
    lower_bound = 1 / fairness_threshold
    upper_bound = fairness_threshold
    for disparity_metric, parity_metric in parity_mapping.items():
        if disparity_metric in fairness_df.columns:
            fairness_df[parity_metric] = fairness_df[disparity_metric].apply(
                lambda x: 'Fair' if pd.isna(x) or (lower_bound <= x <= upper_bound) else 'Unfair'
            )
    if 'Statistical Parity' in fairness_df.columns:
      fairness_df['fairness_conclusion'] = fairness_df['Statistical Parity']
    fairness_summary_df = fairness_df.groupby('attribute_name').agg(
        fairness_conclusion=('fairness_conclusion', lambda x: 'Unfair' if 'Unfair' in x.values else 'Fair')
    ).reset_index()

    # Generar la tabla de equidad por atributo como en el notebook
    f = Fairness()
    fairness_by_attribute_df = f.get_group_attribute_fairness(fairness_df)

    # Generar el gráfico de disparidad inicial por defecto, usando la métrica de disparidad correcta.
    initial_disparity_plot = render_disparity_plot(fairness_df, ['ppr_disparity', 'fdr_disparity'], 'all')

    return {
        "distribution_plots": dist_plots,
        "plots": {
            "disparity_summary": initial_disparity_plot
        },
        "tables": {
            "group_counts": group_counts_df,
            "group_metrics": group_metrics_df,
            "group_metrics_for_plotting": group_metrics_for_plotting,
            "bias_metrics": fairness_df,
            "fairness_summary": fairness_summary_df,
            "fairness_by_attribute": fairness_by_attribute_df
        },
        "metadata": {
            "protected_attributes": protected_attributes,
            "unique_values": {col: df[col].unique().tolist() for col in protected_attributes},
            "fairness_threshold": fairness_threshold
        }
    }
