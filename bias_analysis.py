from aequitas.bias import Bias
import pandas as pd
from aequitas.plotting import Plot
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Create Audit object with the data
from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.fairness import Fairness


def calculate_bias_metrics(xtab, df, ref_groups=None, metric_ref='fpr'):
    """
    Calculate bias metrics using different reference group selection methods.
    
    Args:
        xtab: The crosstab DataFrame from Aequitas Group analysis
        df: Original DataFrame
        ref_groups: Dictionary of reference groups for each attribute
        metric_ref: Metric to use as reference for min_metric calculation
        
    Returns:
        Dictionary containing the bias analysis results
    """
    b = Bias()
    
    results = {
        'predefined': None,
        'major_group': None,
        'min_metric': None,
    }
    try:
        # Calculate disparities with predefined reference groups if provided
        if ref_groups:
            bdf = b.get_disparity_predefined_groups(
                xtab,
                original_df=df,
                ref_groups_dict=ref_groups,
                alpha=0.05,
                mask_significance=True
            )
            results['predefined'] = bdf.round(3).to_dict('records')

        # Calculate disparities using the majority group as reference
        bdfs = b.get_disparity_major_group(
            xtab,
            original_df=df,
            mask_significance=True
        )
        results['major_group'] = bdfs.round(3).to_dict('records')

        # Calculate disparities using the group with minimum error as reference
        bdfmin = b.get_disparity_min_metric(
            xtab,
            original_df=df,
            label_score_ref=metric_ref
        )
        results['min_metric'] = bdfmin.round(3).to_dict('records')

    except Exception as e:
        print(f"Error en el cálculo de métricas de sesgo: {str(e)}")
        return None

    return results

def generate_disparity_plot(plot_type: str, df, xtab, params: dict) -> str:
    """
    Generate disparity plots based on the specified type and parameters.
    
    Args:
        plot_type: Type of plot to generate ('single', 'all_columns', 'all_metrics', 'custom')
        df: DataFrame with the data
        params: Dictionary with plot parameters
            - metric: Metric to plot (for 'single' and 'all_columns')
            - attribute: Attribute to plot (for 'single' and 'all_metrics')
            - metrics: List of metrics to plot (for 'custom')
            - attributes: List of attributes to plot (for 'custom')
            
    Returns:
        Base64 encoded plot image
    """
    #print(f"Generating plot of type: {plot_type}")
    #print(f"Parameters: {params}")
    try:
        aqp = Plot()

        b = Bias()
        bdf = b.get_disparity_predefined_groups(
            xtab,
            original_df=df,
            ref_groups_dict={'raza':'Caucasico', 'Genero':'Hombre', 'categoria_edad':'25 - 45'},  # TODO: Vamos a quemar estos datos pero deben poder enviarese los valores que tenga la otra seccion
            alpha=0.05,
            mask_significance=True
        )

        bdfs =b.get_disparity_major_group(
            xtab,
            original_df=df,
            mask_significance=True
        )
        
        plt.figure(figsize=(12, 6))
        print(f"Plotting type: {plot_type}")

        if plot_type == 'single':
            # Single metric, single attribute
            fig = aqp.plot_disparity(
                bdf,
                group_metric=f"{params['metric']}_disparity",
                attribute_name=params['attribute'],
                significance_alpha=0.05
            )
        
        elif plot_type == 'all_columns':
            # Single metric, all attributes
            fig = aqp.plot_disparity_all(
                bdfs,
                metrics=[f"{params['metric']}_disparity"],
                significance_alpha=0.05
            )
        
        elif plot_type == 'all_metrics':

            bdfmin = b.get_disparity_min_metric(
            xtab, 
            original_df=df, 
            label_score_ref=params['metric'])

            # All metrics, single attribute
            fig = aqp.plot_disparity_all(
                bdfmin,
                attributes=[params['attribute']],
                metrics='all',
                significance_alpha=0.05
            )
        
        elif plot_type == 'custom':
            # Custom selection of metrics and attributes
            metrics = [f"{m}_disparity" for m in params['metrics']]
            fig = aqp.plot_disparity_all(
                bdf,
                metrics=metrics,
                attributes=params.get('attributes', None),
                significance_alpha=0.05
            )

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        print(f"Error generando gráfico: {str(e)}")
        return None
    finally:
        plt.close('all')