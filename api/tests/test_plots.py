"""Smoke tests de las visualizaciones estilo Aequitas (api/plots.py)."""
import os

import pandas as pd
import pytest

from api import metrics_core as mc
from api import plots

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
PROTECTED = ["race", "sex", "age_cat"]


@pytest.fixture(scope="module")
def analysis():
    df = pd.read_csv(os.path.join(FIXTURES, "compas_for_aequitas.csv"))
    return mc.run_full_analysis(df, PROTECTED, "score", "label_value",
                                "majority", {}, 1.25, "fpr")


def _is_png_data_uri(s):
    return isinstance(s, str) and s.startswith("data:image/png;base64,") and len(s) > 1000


@pytest.mark.parametrize("metric", ["fpr", "tpr", "precision", "ppr"])
def test_group_metric_plot(analysis, metric):
    gm = analysis["tables"]["group_metrics_for_plotting"]
    assert _is_png_data_uri(plots.render_group_metric_plot(gm, metric, "all"))
    assert _is_png_data_uri(plots.render_group_metric_plot(gm, metric, "race"))


@pytest.mark.parametrize("metric", ["fpr_disparity", "tpr_disparity", "npv_disparity"])
def test_disparity_treemap(analysis, metric):
    bm = analysis["tables"]["bias_metrics"]
    assert _is_png_data_uri(plots.render_disparity_treemap(bm, metric, "all"))
    assert _is_png_data_uri(plots.render_disparity_treemap(bm, [metric], "sex"))


def test_plots_handle_missing_metric(analysis):
    gm = analysis["tables"]["group_metrics_for_plotting"]
    assert plots.render_group_metric_plot(gm, "no_existe", "all") == ""
    assert plots.render_disparity_treemap(analysis["tables"]["bias_metrics"], "no_existe", "all") == ""
