from aequitas.fairness import Fairness
from aequitas.bias import Bias
import pandas as pd

def calculate_fairness_metrics(bdf, xtab, df):
    """
    Calculate fairness metrics using Aequitas.
    
    Args:
        bdf: Bias DataFrame from previous analysis
        xtab: Crosstab DataFrame from Group analysis
        df: Original DataFrame
        
    Returns:
        Dictionary containing fairness metrics and determinations
    """
    try:

        if bdf is None:
            raise ValueError("Input DataFrames cannot be None")
        b = Bias()
        
        # Calculate group fairness metrics
        bdf = b.get_disparity(xtab, original_df=df, mask_significance=True)
        
        f = Fairness()
        
        # Calculate fairness metrics for each group
        fdf = f.get_group_value_fairness(bdf)
        
        # Get list of all fairness determinations
        parity_determinations = f.list_parities(fdf)
        
        # Get absolute metrics for reference
        absolute_metrics = ['tpr', 'tnr', 'for', 'fdr', 'fpr', 'fnr', 'npv', 'precision', 'ppr', 'pprev', 'prev']
        
        # Get fairness determinations for each attribute
        gaf = f.get_group_attribute_fairness(fdf)
        
        # Prepare the response data
        fairness_metrics = {
            # Detailed fairness metrics for each group
            'group_value_fairness': fdf[['attribute_name', 'attribute_value'] + absolute_metrics + parity_determinations].to_dict('records'),
            
            # Overall fairness determinations for each attribute
            'group_attribute_fairness': gaf.to_dict()
        }
        
        return fairness_metrics
        
    except Exception as e:
        print(f"Error calculating fairness metrics: {str(e)}")
        return None