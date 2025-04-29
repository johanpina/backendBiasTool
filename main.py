from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union, List
from typing import List, Dict, Optional
import pandas as pd
import json
from aequitas import Audit
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot
import aequitas.plot as ap

import tempfile
import os
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bias_analysis import calculate_bias_metrics, generate_disparity_plot

app = FastAPI(title="API de Análisis de Sesgos")

# Diccionario de traducciones para las métricas
metric_translations = {
    'attribute_name': 'Variable Protegida',
    'attribute_value': 'Valor',
    'tpr': 'Tasa de Verdaderos Positivos',
    'tnr': 'Tasa de Verdaderos Negativos',
    'for': 'Tasa de Falsos Omitidos',
    'fdr': 'Tasa de Falsos Descubrimientos',
    'fpr': 'Tasa de Falsos Positivos',
    'fnr': 'Tasa de Falsos Negativos',
    'npv': 'Valor Predictivo Negativo',
    'precision': 'Precisión',
    'ppr': 'Tasa de Predicción Positiva',
    'pprev': 'Prevalencia Predicha',
    'prev': 'Prevalencia Real',
    'accuracy': 'Exactitud',
    'group_label_pos': 'Etiqueta Positiva del Grupo',
    'group_label_neg': 'Etiqueta Negativa del Grupo',
    'model_id': 'ID del Modelo',
    'score_threshold': 'Umbral de Puntaje',
    'total_entities': 'Total de Entidades',
}

# Variable global para almacenar el último análisis
last_analysis = None
last_bias_analysis = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ColumnSelection(BaseModel):
    predictions: str
    actual: str
    protected: List[str]

class PreviewResponse(BaseModel):
    columns: List[str]
    rows: List[Dict]

class AnalysisResponse(BaseModel):
    metrics: List[Dict]
    protected_attributes: List[str]
    initial_plot: str

def prepare_data(file: UploadFile, column_selection: Dict) -> tuple:
    """Prepara los datos para el análisis."""
    df = pd.read_csv(file.file)
    
    df_renamed = df.copy()
    df_renamed['score'] = df[column_selection['predictions']]
    df_renamed['label_value'] = df[column_selection['actual']]
    df_renamed['entity_id'] = range(len(df))

    g = Group()
    xtab, _ = g.get_crosstabs(df_renamed)
    
    unique_values = {
        col: df[col].unique().tolist()
        for col in column_selection['protected']
    }
    
    return df_renamed, xtab, unique_values

def generate_plot(xtab, metric: str, attribute: str):
    try:
        aqp = Plot()
        
        # Asegurarse de que xtab es un DataFrame
        if not isinstance(xtab, pd.DataFrame):
            raise ValueError("Los datos de análisis no son válidos")
            
        # Verificar que la métrica existe en el DataFrame
        if metric not in xtab.columns:
            raise ValueError(f"La métrica {metric} no está disponible")

        # Si se especifica un atributo y no es "todas", filtrar el DataFrame
        plot_data = xtab
        if attribute and attribute.lower() != "todas":
            plot_data = xtab[xtab['attribute_name'] == attribute].copy()
            if plot_data.empty:
                raise ValueError(f"No hay datos para el atributo {attribute}")

        # Crear una nueva figura
        plt.figure(figsize=(10, 6))
        
        # Generar el gráfico
        fig = aqp.plot_group_metric(plot_data, metric, title=False)

        # 🪄 Aquí traducimos los textos de matplotlib
        ax = plt.gca()
        translations = {
            "Group Value": "Valor del Grupo",
            "Group": "Grupo",
            "Value": "Valor",
            "Disparity": "Disparidad",
            "False Negative Rate": "Tasa de Falsos Negativos",
            "False Positive Rate": "Tasa de Falsos Positivos",
            "True Positive Rate": "Tasa de Verdaderos Positivos",
            "True Negative Rate": "Tasa de Verdaderos Negativos",
            "Accuracy": "Precisión",
            "Label Value": "Valor Real",
            "Score": "Predicción",
            "Absolute Metric Magnitude":"Magnitud Absoluta de la Métrica",
        }

        # Traducir ejes y título
        if ax.get_xlabel() in translations:
            ax.set_xlabel(translations[ax.get_xlabel()])
        if ax.get_ylabel() in translations:
            ax.set_ylabel(translations[ax.get_ylabel()])
        if ax.get_title() in translations:
            ax.set_title(translations[ax.get_title()])

        # Traducir las etiquetas de las barras
        xticklabels = [translations.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()]
        ax.set_xticklabels(xticklabels)
        
        ax.set_title(f"Métrica: {metric_translations[metric]}")

        # Traducir leyenda si existe
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_text(translations.get(text.get_text(), text.get_text()))
        
        # Guardar la figura en un buffer de memoria
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close('all')
        buf.seek(0)
        
        # Convertir a base64
        plot_base64 = base64.b64encode(buf.getvalue()).decode()
        return plot_base64
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al generar el gráfico: {str(e)}")
    finally:
        plt.close('all')

@app.get("/api/plot/{metric}")
async def get_plot(metric: str, attribute: str):
    global last_analysis
    
    try:
        if last_analysis is None:
            raise HTTPException(
                status_code=400,
                detail="No hay análisis disponible. Por favor, realice primero un análisis."
            )
        print(f"Generando gráfico para la métrica: {metric} y atributo: {attribute}")
        plot_base64 = generate_plot(last_analysis, metric, attribute)
        return JSONResponse(content={"plot": plot_base64})

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/preview", response_model=PreviewResponse)
async def preview_data(file: UploadFile):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")
        
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            df = pd.read_csv(tmp_path)
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="El archivo CSV está vacío")
        except pd.errors.ParserError:
            raise HTTPException(status_code=400, detail="Error al parsear el archivo CSV. Verifique el formato")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="El archivo no contiene datos")
        
        preview_rows = df.head().to_dict('records')
        
        for row in preview_rows:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
        
        response_data = {
            "columns": df.columns.tolist(),
            "rows": preview_rows
        }
        
        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.post("/api/analyze")
async def analyze_data(file: UploadFile, columns: str = Form(...),params: Optional[str] = Form(None)):
    global last_analysis
    try:
        try:
            column_selection = json.loads(columns)
            analysis_params = json.loads(params) if params else {
            "metric_ref": "fpr",
            "ref_groups": {}
        }
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Error al parsear la selección de columnas")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            df = pd.read_csv(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {str(e)}")
        
        required_columns = {
            column_selection['predictions'],
            column_selection['actual']
        }
        
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail="Las columnas seleccionadas no existen en el archivo"
            )

        if not column_selection['protected']:
            raise HTTPException(
                status_code=400,
                detail="Debe seleccionar al menos una variable protegida"
            )
        # TODO: validar que se cambie el nombre de la columna de prediccion y la real 
        # TODO: garantizar que no se dupliquen las columnas cuando no tiene el mismo nombre.
        df_renamed = df.copy()
        df_renamed['score'] = df[column_selection['predictions']]
        df_renamed['label_value'] = df[column_selection['actual']]
        df_renamed['entity_id'] = range(len(df))

        g = Group()
        
        try:
            xtab, _ = g.get_crosstabs(df_renamed)
            last_analysis = xtab
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error al calcular métricas: {str(e)}"
            )


        absolute_metrics = g.list_absolute_metrics(xtab)

        # Tabla total de instancias por cada subgrupo
        total_instance_per_group = xtab[[col for col in xtab.columns if col not in absolute_metrics]]
        
        # Reemplazar lo que diga binary por binario en la columna de score threshold
        total_instance_per_group['score_threshold'] = total_instance_per_group['score_threshold'].replace('binary 0/1', 'binario 0/1')
        total_instance_per_group.rename(columns=metric_translations, inplace=True)
        metrics_df = xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(2)

        metrics_df = metrics_df.rename(columns=metric_translations)
        metrics = metrics_df.to_dict('records')
        # Reemplazar NaN por None

        for metric in metrics:
            for key, value in metric.items():
                if pd.isna(value):
                    metric[key] = None

        initial_plot = generate_plot(xtab, 'fnr', "todas")
        # Agregar "Todas" a la lista de atributos protegidos
        protected_attributes = ['todas'] + column_selection['protected']

        # Get unique values for each protected attribute
        unique_values = {
            col: df[col].unique().tolist()
            for col in column_selection['protected']
        }

        # Calculate bias metrics with parameters
        bias_metrics = calculate_bias_metrics(
            xtab,
            df_renamed,
            ref_groups=analysis_params["ref_groups"],
            metric_ref=analysis_params["metric_ref"]
        )


        response_data = {
            "metrics": metrics,
            "instance_per_subgroup": total_instance_per_group.to_dict('records'),
            "protected_attributes": column_selection['protected'],
            "initial_plot": initial_plot,
            "bias_metrics": bias_metrics,
            "unique_values": unique_values,
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Error en el análisis de datos: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass

def analyze_fairness(bdf):
    last_bias_analysis = bdf
    

    if last_bias_analysis is None:
        raise HTTPException(
            status_code=400,
            detail="No hay análisis disponible. Por favor, realice primero un análisis."
        )
    
    f = Fairness()
    fdf = f.get_group_value_fairness(last_bias_analysis)
    parity_determinations = f.list_parities(fdf)
    gaf = f.get_group_attribute_fairness(fdf)
            
    # Reemplazar lo que diga binary por binario en la columna de score threshold
    gaf['score_threshold'] = gaf['score_threshold'].replace('binary 0/1', 'binario 0/1')
    gaf.rename(columns=metric_translations, inplace=True)

    metrics = gaf.to_dict('records')
    return metrics
    


@app.post("/api/analyze_bias")
async def analyze_bias(
    file: UploadFile,
    columns: str = Form(...),
    params: Optional[str] = Form(None)
):
    global last_bias_analysis

    try:
        column_selection = json.loads(columns)
        analysis_params = json.loads(params) if params else {}

        # Extraer parámetros
        ref_method = analysis_params.get("referenceMethod", "minority")
        ref_groups = analysis_params.get("referenceGroups", {})
        metric_ref = analysis_params.get("metric_ref", "fpr")  # usado solo con "minority"
        disparity_tolerance = analysis_params.get("disparity_tolerance", 1.25)  # usado solo con "minority"
        print(f"Referencia: {ref_method}, Grupos: {ref_groups}, Métrica: {metric_ref}, Tolerancia: {disparity_tolerance}")

        # Cargar el CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        df = pd.read_csv(tmp_path)

        # Validar columnas requeridas
        required_columns = {
            column_selection['predictions'],
            column_selection['actual']
        }

        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail="Las columnas seleccionadas no existen en el archivo"
            )

        if not column_selection['protected']:
            raise HTTPException(
                status_code=400,
                detail="Debe seleccionar al menos una variable protegida"
            )

        # Renombrar columnas para Aequitas
        df_renamed = df.copy()
        df_renamed['score'] = df[column_selection['predictions']]
        df_renamed['label_value'] = df[column_selection['actual']]
        df_renamed['entity_id'] = range(len(df))

        g = Group()
        xtab, _ = g.get_crosstabs(df_renamed)
        

        # Obtener métricas absolutas
        absolute_metrics = g.list_absolute_metrics(xtab)

        # Tabla de instancias por grupo
        total_instance_per_group = xtab[[col for col in xtab.columns if col not in absolute_metrics]]
        total_instance_per_group['score_threshold'] = total_instance_per_group['score_threshold'].replace(
            'binary 0/1', 'binario 0/1'
        )
        total_instance_per_group.rename(columns=metric_translations, inplace=True)

        # Tabla de métricas absolutas
        metrics_df = xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(2)
        metrics_df.rename(columns=metric_translations, inplace=True)
        metrics = metrics_df.to_dict('records')
        for metric in metrics:
            for k, v in metric.items():
                if pd.isna(v):
                    metric[k] = None

        # Valores únicos por atributo protegido
        unique_values = {
            col: df[col].unique().tolist()
            for col in column_selection['protected']
        }

        # Calcular disparidades según método
        b = Bias()
        if ref_method == "minority":
            bias_df = b.get_disparity_min_metric(
                xtab,
                original_df=df_renamed,
                label_score_ref=metric_ref
            )
        elif ref_method == "majority":
            bias_df = b.get_disparity_major_group(
                xtab,
                original_df=df_renamed,
                mask_significance=True
            )
        elif ref_method == "custom":
            bias_df = b.get_disparity_predefined_groups(
                xtab,
                original_df=df_renamed,
                ref_groups_dict=ref_groups,
                alpha=0.05,
                mask_significance=True
            )
        else:
            raise HTTPException(status_code=400, detail=f"Método de referencia no reconocido: {ref_method}")
        
    
        # Convertir a JSON
        bias_metrics = bias_df.rename(columns=metric_translations).replace(
            'binary 0/1', 'binario 0/1'
        ).round(3)
        last_bias_analysis = bias_df
        plot = generate_bias_plot(bias_df, metrics='all', attributes=[])

        try:
                
            audit = Audit(df_renamed.drop(columns='entity_id'), label_column=column_selection['actual'])
            audit.audit(bias_args={
                "alpha": 0.05,
                "check_significance": True,
                "mask_significance": True
            })
            dispa_plot = audit.disparity_plot(metrics=[metric_ref], attribute=column_selection['protected'][0], fairness_threshold=disparity_tolerance)
            print("finalicé disparity plot")
            dispa_err_plot = ap.absolute(bias_df, metrics_list=[metric_ref], attribute=column_selection['protected'][0], fairness_threshold = disparity_tolerance)
            print("finalicé disparity plot dinamic")
        except Exception as e:
            print(f"Error al generar el gráfico dinámico de disparidad: {str(e)}")
        
        fairness_metrics = analyze_fairness(bias_df)

        #print("Plot:", plot)

        response_data = {
            "protected_attributes": column_selection['protected'],
            "unique_values": unique_values,
            "group_metrics": metrics,
            "bias_metrics": bias_metrics.to_dict('records'),
            "bias_plot": plot,
            "fairness_metrics": fairness_metrics,
        }
        #print("Response data:", response_data)
        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Error en el análisis de sesgo: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass



def generate_bias_plot(xtab, metrics, attributes):
    try:
        aqp = Plot()
        
        # Asegurarse de que xtab es un DataFrame
        if not isinstance(xtab, pd.DataFrame):
            raise ValueError("Los datos de análisis no son válidos")
            
        # Verificar que la métrica existe en el DataFrame
        #if metric not in xtab.columns:
        #    raise ValueError(f"La métrica {metric} no está disponible")

        
        # Si se especifica un atributo y no es "todas", filtrar el DataFrame
        plot_data = xtab
        #print("Plot data:", plot_data) 
        if metrics == "todas":
            plot_metrics = 'all'
        else:
            plot_metrics = metrics
        

        # Crear una nueva figura
        plt.figure(figsize=(10, 6))
        
        # Generar el gráfico
        fig = aqp.plot_disparity_all(plot_data, metrics=metrics, attributes=attributes, significance_alpha=0.05)

        # 🪄 Aquí traducimos los textos de matplotlib
        ax = plt.gca()
        translations = {
            "Group Value": "Valor del Grupo",
            "Group": "Grupo",
            "Value": "Valor",
            "Disparity": "Disparidad",
            "False Negative Rate": "Tasa de Falsos Negativos",
            "False Positive Rate": "Tasa de Falsos Positivos",
            "True Positive Rate": "Tasa de Verdaderos Positivos",
            "True Negative Rate": "Tasa de Verdaderos Negativos",
            "Accuracy": "Precisión",
            "Label Value": "Valor Real",
            "Score": "Predicción",
            "Absolute Metric Magnitude":"Magnitud Absoluta de la Métrica",
        }

        # Traducir ejes y título
        if ax.get_xlabel() in translations:
            ax.set_xlabel(translations[ax.get_xlabel()])
        if ax.get_ylabel() in translations:
            ax.set_ylabel(translations[ax.get_ylabel()])
        if ax.get_title() in translations:
            ax.set_title(translations[ax.get_title()])

        # Traducir las etiquetas de las barras
        xticklabels = [translations.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()]
        ax.set_xticklabels(xticklabels)
        
        #ax.set_title(f"Métrica: {metric_translations[metric]}")

        # Traducir leyenda si existe
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_text(translations.get(text.get_text(), text.get_text()))
        
        # Guardar la figura en un buffer de memoria
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        #plt.savefig('test.png', format='png', bbox_inches='tight', dpi=300)
        plt.close('all')
        buf.seek(0)
        
        # Convertir a base64
        plot_base64 = base64.b64encode(buf.getvalue()).decode()
        return plot_base64
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al generar el gráfico: {str(e)}")
    finally:
        plt.close('all')
    

class PlotRequest(BaseModel):
    metrics: Union[str, List[str]]
    attributes: List[str]

@app.post("/api/bias_plot")
async def get_plot(request: PlotRequest):
    global last_bias_analysis

    if last_bias_analysis is None:
        raise HTTPException(
            status_code=400,
            detail="No hay análisis disponible. Por favor, realice primero un análisis."
        )

    # 🧼 Normalizamos: si metrics es un string lo convertimos a lista
    if isinstance(request.metrics, str):
        metrics = request.metrics
    else:
        metrics = request.metrics

    print(f"Generando gráfico para la métrica: {metrics} y atributo: {request.attributes}")
    
    plot_base64 = generate_bias_plot(
        last_bias_analysis,
        metrics=metrics,
        attributes=request.attributes
    )
    return JSONResponse(content={"plot": plot_base64})