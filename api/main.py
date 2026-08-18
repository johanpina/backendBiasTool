from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import json
import tempfile
import os
import chardet

from .bias_analysis import run_full_analysis, render_disparity_plot, render_absolute_plot

app = FastAPI(title="API de Análisis de Sesgos y Equidad")


def to_records(df: pd.DataFrame) -> List[Dict]:
    """Convierte un DataFrame a lista de dicts sustituyendo NaN por None.

    JSON estándar no admite el token ``NaN``; ``fetch().json()`` del navegador
    falla si aparece. Esto garantiza una respuesta parseable con cualquier
    dataset (p. ej. subgrupos pequeños que producen métricas NaN).
    """
    clean = df.astype(object).where(pd.notna(df), None)
    return clean.to_dict(orient="records")


def serialize_tables(tables: Dict) -> Dict:
    """Serializa el dict de tablas del contrato a listas de dicts (con NaN->None).

    Traduce al español solo las tablas que el frontend muestra directamente;
    ``group_metrics_for_plotting`` y ``bias_metrics`` van crudas (en inglés)
    porque el frontend deriva de ellas los nombres de métricas. Reutilizable para
    las tablas globales y las de cada clase (multiclase).
    """
    return {
        "group_counts": to_records(translate_df(tables["group_counts"])),
        "group_metrics": to_records(translate_df(tables["group_metrics"])),
        "group_metrics_for_plotting": to_records(tables["group_metrics_for_plotting"]),
        "bias_metrics": to_records(tables["bias_metrics"]),
        "fairness_summary": to_records(translate_df(tables["fairness_summary"])),
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    """Verificación de estado del servicio."""
    return {"status": "ok"}


def read_csv_robust(filepath: str, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath, encoding='utf-8', **kwargs)
    except (UnicodeDecodeError, pd.errors.ParserError):
        try:
            return pd.read_csv(filepath, encoding='latin1', **kwargs)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo CSV. Error: {e}")

def translate_df(df: pd.DataFrame) -> pd.DataFrame:
    metric_translations = {
        'class': 'Clase',
        'model_id': 'ID Modelo', 'score_threshold': 'Umbral Score', 'k': 'k',
        'attribute_name': 'Atributo', 'attribute_value': 'Valor Atributo',
        'pp': 'Predichos Positivos', 'pn': 'Predichos Negativos', 'fp': 'Falsos Positivos',
        'fn': 'Falsos Negativos', 'tn': 'Verdaderos Negativos', 'tp': 'Verdaderos Positivos',
        'group_label_pos': 'Etiquetas Positivas Grupo', 'group_label_neg': 'Etiquetas Negativas Grupo',
        'group_size': 'Tamaño Grupo', 'total_entities': 'Total Entidades',
        'prev': 'Prevalencia', 'pprev': 'Prevalencia Predicha', 'accuracy': 'Exactitud',
        'tpr': 'Tasa Verdaderos Positivos (TPR)', 'tnr': 'Tasa Verdaderos Negativos (TNR)',
        'for': 'Tasa Falsas Omisiones (FOR)', 'fdr': 'Tasa Falsos Descubrimientos (FDR)',
        'fpr': 'Tasa Falsos Positivos (FPR)', 'fnr': 'Tasa Falsos Negativos (FNR)',
        'npv': 'Valor Predictivo Negativo (NPV)', 'precision': 'Precisión',
        'ppr': 'Proporción Predicha Positiva (PPR)',
        'ppr_disparity': 'Disparidad PPR', 'pprev_disparity': 'Disparidad PPREV',
        'precision_disparity': 'Disparidad Precisión', 'fdr_disparity': 'Disparidad FDR',
        'for_disparity': 'Disparidad FOR', 'fpr_disparity': 'Disparidad FPR',
        'fnr_disparity': 'Disparidad FNR', 'tpr_disparity': 'Disparidad TPR',
        'tnr_disparity': 'Disparidad TNR', 'npv_disparity': 'Disparidad NPV',
        'accuracy_disparity': 'Disparidad Exactitud', 'fairness_conclusion': 'Conclusión Equidad',
        'Statistical Parity': 'Paridad Estadística', 'Impact Parity': 'Paridad de Impacto',
        'FDR Parity': 'Paridad FDR', 'FOR Parity': 'Paridad FOR', 'FPR Parity': 'Paridad FPR',
        'FNR Parity': 'Paridad FNR', 'TPR Parity': 'Paridad TPR', 'TNR Parity': 'Paridad TNR',
        'NPV Parity': 'Paridad NPV', 'Precision Parity': 'Paridad de Precisión', 'Accuracy Parity': 'Paridad de Exactitud'
    }
    return df.rename(columns=metric_translations)

@app.post("/api/preview")
async def preview_file(file: UploadFile = Form(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        df = read_csv_robust(tmp_path, nrows=5)
        return JSONResponse(content={"columns": df.columns.tolist(), "preview": df.to_dict(orient='records')})
    finally:
        os.unlink(tmp_path)


@app.post("/api/eda")
async def eda_endpoint(file: UploadFile = Form(...), params: Optional[str] = Form(None)):
    """Análisis exploratorio: perfil de columnas, asociaciones y alertas de sesgo."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV.")
    try:
        eda_params = json.loads(params) if params else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Parámetros JSON no válidos.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        from .eda import run_eda
        df = read_csv_robust(tmp_path)
        result = run_eda(df, min_group_size=int(eda_params.get("minGroupSize", 50)))
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en el análisis exploratorio: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/api/full_analysis")
async def full_analysis(file: UploadFile = Form(...), columns: str = Form(...), params: Optional[str] = Form(None)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV.")
    try:
        column_selection = json.loads(columns)
        analysis_params = json.loads(params) if params else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Parámetros JSON no válidos.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        df = read_csv_robust(tmp_path)
        analysis_results = run_full_analysis(
            df=df,
            protected_attributes=column_selection.get('protected', []),
            score_col=column_selection.get('predictions'),
            label_col=column_selection.get('actual'),
            ref_method=analysis_params.get("referenceMethod", "majority"),
            ref_groups=analysis_params.get("referenceGroups", {}),
            # Acepta ambas convenciones de nombre para el umbral de tolerancia.
            fairness_threshold=float(
                analysis_params.get("fairnessThreshold",
                                    analysis_params.get("fairness_threshold", 1.25))
            ),
            performance_metric=analysis_params.get("performanceMetric", "fpr"),
            min_group_size=int(analysis_params.get("minGroupSize",
                                                   analysis_params.get("min_group_size", 50))),
        )
        response = {
            "distribution_plots": analysis_results["distribution_plots"],
            "plots": analysis_results["plots"],
            "tables": serialize_tables(analysis_results["tables"]),
            "metadata": analysis_results["metadata"]
        }
        # Multiclase: adjuntar tablas por clase (forma binaria) y resumen global.
        if "by_class" in analysis_results:
            response["by_class"] = {
                cls: {
                    "tables": serialize_tables(entry["tables"]),
                    "plots": entry.get("plots", {}),
                }
                for cls, entry in analysis_results["by_class"].items()
            }
            response["fairness_overall"] = to_records(
                translate_df(analysis_results["fairness_overall"])
            )
        return JSONResponse(content=response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno durante el análisis: {str(e)}")
    finally:
        os.unlink(tmp_path)

@app.post("/api/rerender_plot")
async def rerender_plot_endpoint(payload: Dict):
    try:
        bias_df = pd.DataFrame(payload.get('bias_metrics'))
        metrics = payload.get('metrics')
        attributes = payload.get('attributes')
        
        # Usamos el primer atributo de la lista (o 'all' si está vacía)
        attribute_to_plot = attributes[0] if attributes else 'all'
        
        plot_base64 = render_disparity_plot(bias_df, metrics, attribute_to_plot)
        return JSONResponse(content={"plot": plot_base64})
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/absolute_plot")
async def absolute_plot_endpoint(payload: Dict):
    try:
        group_metrics_df = pd.DataFrame(payload.get('group_metrics_for_plotting'))
        metric = payload.get('metric')
        attribute = payload.get('attribute')
        plot_base64 = render_absolute_plot(group_metrics_df, metric, attribute)
        return JSONResponse(content={"plot": plot_base64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/group_metric_plot")
async def group_metric_plot_endpoint(payload: Dict):
    try:
        # El frontend envía 'group_metrics', lo convertimos a DataFrame
        group_metrics_df = pd.DataFrame(payload.get('group_metrics'))
        metric = payload.get('metric')
        attribute = payload.get('attribute')
        
        # Reutilizamos la función de ploteo existente que hace lo que necesitamos
        plot_base64 = render_absolute_plot(group_metrics_df, metric, attribute)
        
        return JSONResponse(content={"plot": plot_base64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recalculate_fairness")
async def recalculate_fairness_endpoint(payload: Dict):
    try:
        bias_df = pd.DataFrame(payload.get('bias_metrics'))
        fairness_threshold = float(payload.get('fairnessThreshold', 1.25))
        min_group_size = int(payload.get('minGroupSize', 50))

        # Importar la función necesaria
        from .bias_analysis import recalculate_fairness

        updated_tables = recalculate_fairness(bias_df, fairness_threshold, min_group_size)
        
        response = {
            "fairness_summary": to_records(translate_df(updated_tables["fairness_summary"])),
            "fairness_by_attribute": to_records(updated_tables["fairness_by_attribute"])
        }
        return JSONResponse(content=response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))