"""Tests del motor ligero metrics_core.

Estrategia de validación:
  1. Golden COMPAS: comparar contra fixtures generados con Aequitas 1.0.0 real
     (group_metrics, group_counts, disparidades con 3 estrategias de referencia).
  2. Fairlearn: validación cruzada independiente de las tasas por grupo.
  3. Contrato de columnas, estrategias de referencia e idempotencia.

Los fixtures golden viven en api/tests/fixtures/golden_*.csv.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from api import metrics_core as mc

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
PROTECTED = ["race", "sex", "age_cat"]
KEYS = ["attribute_name", "attribute_value"]


@pytest.fixture(scope="module")
def compas():
    return pd.read_csv(os.path.join(FIXTURES, "compas_for_aequitas.csv"))


@pytest.fixture(scope="module")
def golden_meta():
    with open(os.path.join(FIXTURES, "golden_meta.json")) as f:
        return json.load(f)


def _sorted(df):
    return (
        df.copy()
        .assign(attribute_value=lambda d: d["attribute_value"].astype(str))
        .sort_values(KEYS)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 1. Golden COMPAS: métricas absolutas y conteos
# ---------------------------------------------------------------------------
def test_group_counts_match_aequitas(compas):
    xtab, _ = mc.get_crosstabs(compas[PROTECTED + ["score", "label_value"]], PROTECTED)
    golden = pd.read_csv(os.path.join(FIXTURES, "golden_group_counts.csv"))

    count_cols = ["pp", "pn", "fp", "fn", "tn", "tp",
                  "group_label_pos", "group_label_neg", "group_size", "total_entities"]
    mine = _sorted(xtab[KEYS + count_cols])
    gold = _sorted(golden[KEYS + count_cols])
    assert_frame_equal(mine[count_cols], gold[count_cols], check_dtype=False)


def test_group_metrics_match_aequitas(compas):
    xtab, absolute_metrics = mc.get_crosstabs(
        compas[PROTECTED + ["score", "label_value"]], PROTECTED)
    golden = pd.read_csv(os.path.join(FIXTURES, "golden_group_metrics.csv"))

    mine = _sorted(xtab[KEYS + absolute_metrics])
    gold = _sorted(golden[KEYS + absolute_metrics])
    for m in absolute_metrics:
        np.testing.assert_allclose(
            mine[m].to_numpy(dtype=float), gold[m].to_numpy(dtype=float),
            rtol=1e-9, atol=1e-9, equal_nan=True,
            err_msg=f"Métrica absoluta '{m}' difiere de Aequitas")


@pytest.mark.parametrize("ref_method,ref_groups,perf,golden_file", [
    ("custom", {"race": "Caucasian", "sex": "Male", "age_cat": "25 - 45"}, "fpr",
     "golden_disparity_custom.csv"),
    ("majority", {}, "fpr", "golden_disparity_majority.csv"),
    ("best_performance", {}, "fpr", "golden_disparity_minmetric.csv"),
])
def test_disparities_match_aequitas(compas, ref_method, ref_groups, perf, golden_file):
    xtab, _ = mc.get_crosstabs(compas[PROTECTED + ["score", "label_value"]], PROTECTED)
    if ref_method == "best_performance":
        bias_df = mc.get_disparity_min_metric(xtab, mask_significance=False)
    else:
        ref_map = mc._reference_rows(xtab, ref_method, ref_groups, perf)
        bias_df = mc.get_disparity(xtab, ref_map, mask_significance=False)

    golden = pd.read_csv(os.path.join(FIXTURES, golden_file))
    disp_cols = [c for c in golden.columns if c.endswith("_disparity")]

    mine = _sorted(bias_df[KEYS + disp_cols])
    gold = _sorted(golden[KEYS + disp_cols])
    for c in disp_cols:
        np.testing.assert_allclose(
            mine[c].to_numpy(dtype=float), gold[c].to_numpy(dtype=float),
            rtol=1e-9, atol=1e-9, equal_nan=True,
            err_msg=f"Disparidad '{c}' difiere de Aequitas ({ref_method})")


# ---------------------------------------------------------------------------
# 2. Validación cruzada con Fairlearn (oráculo independiente)
# ---------------------------------------------------------------------------
def test_rates_match_fairlearn(compas):
    from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate
    from sklearn.metrics import precision_score, accuracy_score

    xtab, _ = mc.get_crosstabs(compas[PROTECTED + ["score", "label_value"]], PROTECTED)
    y_true = compas["label_value"].astype(int)
    y_pred = compas["score"].astype(int)

    for attr in PROTECTED:
        mf = MetricFrame(
            metrics={
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
                "precision": lambda yt, yp: precision_score(yt, yp, zero_division=np.nan),
                "accuracy": accuracy_score,
                "selection_rate": lambda yt, yp: np.mean(yp),
            },
            y_true=y_true, y_pred=y_pred, sensitive_features=compas[attr],
        )
        by_group = mf.by_group
        sub = xtab[xtab["attribute_name"] == attr].set_index("attribute_value")
        for grp in by_group.index:
            row = sub.loc[str(grp)]
            fl = by_group.loc[grp]
            np.testing.assert_allclose(row["tpr"], fl["tpr"], rtol=1e-9, atol=1e-9, equal_nan=True)
            np.testing.assert_allclose(row["fpr"], fl["fpr"], rtol=1e-9, atol=1e-9, equal_nan=True)
            np.testing.assert_allclose(row["accuracy"], fl["accuracy"], rtol=1e-9, atol=1e-9, equal_nan=True)
            np.testing.assert_allclose(row["pprev"], fl["selection_rate"], rtol=1e-9, atol=1e-9, equal_nan=True)
            if not np.isnan(fl["precision"]):
                np.testing.assert_allclose(row["precision"], fl["precision"], rtol=1e-9, atol=1e-9)


def test_rate_identities(compas):
    """Identidades matemáticas: fnr=1-tpr, tnr=1-fpr, fdr=1-precision, for=1-npv."""
    xtab, _ = mc.get_crosstabs(compas[PROTECTED + ["score", "label_value"]], PROTECTED)
    valid = xtab.dropna(subset=["tpr", "fpr", "precision", "npv"])
    np.testing.assert_allclose(valid["fnr"], 1 - valid["tpr"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(valid["tnr"], 1 - valid["fpr"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(valid["fdr"], 1 - valid["precision"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(valid["for"], 1 - valid["npv"], rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. Estrategias de referencia
# ---------------------------------------------------------------------------
def test_reference_majority_and_minority(compas):
    xtab, _ = mc.get_crosstabs(compas[PROTECTED + ["score", "label_value"]], PROTECTED)
    maj = mc._reference_rows(xtab, "majority", {}, "fpr")
    mino = mc._reference_rows(xtab, "minority", {}, "fpr")
    for attr in PROTECTED:
        sub = xtab[xtab["attribute_name"] == attr]
        assert maj[attr] == sub.loc[sub["group_size"].idxmax(), "attribute_value"]
        assert mino[attr] == sub.loc[sub["group_size"].idxmin(), "attribute_value"]


def test_reference_row_has_unit_disparity(compas):
    xtab, _ = mc.get_crosstabs(compas[PROTECTED + ["score", "label_value"]], PROTECTED)
    ref_map = mc._reference_rows(xtab, "majority", {}, "fpr")
    bias_df = mc.get_disparity(xtab, ref_map, mask_significance=False)
    for attr, ref_val in ref_map.items():
        row = bias_df[(bias_df["attribute_name"] == attr) &
                      (bias_df["attribute_value"] == ref_val)]
        for m in mc.DISPARITY_METRICS:
            val = row[f"{m}_disparity"].iloc[0]
            # La referencia tiene disparidad 1.0 (o NaN si su métrica es NaN).
            assert np.isnan(val) or abs(val - 1.0) < 1e-12


def test_custom_reference_invalid_raises(compas):
    xtab, _ = mc.get_crosstabs(compas[PROTECTED + ["score", "label_value"]], PROTECTED)
    with pytest.raises(ValueError):
        mc._reference_rows(xtab, "custom", {"race": "NoExiste"}, "fpr")


# ---------------------------------------------------------------------------
# 4. Contrato de columnas e idempotencia
# ---------------------------------------------------------------------------
def test_column_contract(compas):
    res = mc.run_full_analysis(compas, PROTECTED, "score", "label_value",
                               "majority", {}, 1.25, "fpr")
    tables = res["tables"]

    expected_counts = {"model_id", "score_threshold", "k", "attribute_name",
                       "attribute_value", "pp", "pn", "fp", "fn", "tn", "tp",
                       "group_label_pos", "group_label_neg", "group_size", "total_entities"}
    assert expected_counts.issubset(set(tables["group_counts"].columns))
    assert set(mc.ABSOLUTE_METRICS).issubset(set(tables["group_metrics"].columns))

    bias_cols = set(tables["bias_metrics"].columns)
    for m in mc.DISPARITY_METRICS:
        assert f"{m}_disparity" in bias_cols
    assert "fairness_conclusion" in bias_cols
    assert set(tables["fairness_summary"].columns) == {"attribute_name", "fairness_conclusion"}
    assert res["metadata"]["task_type"] == "binary"


def test_recalculate_fairness_idempotent(compas):
    res = mc.run_full_analysis(compas, PROTECTED, "score", "label_value",
                               "majority", {}, 1.5, "fpr")
    bias_df = res["tables"]["bias_metrics"]
    recalc = mc.recalculate_fairness(bias_df, 1.5)
    assert_frame_equal(
        res["tables"]["fairness_summary"].reset_index(drop=True),
        recalc["fairness_summary"].reset_index(drop=True),
    )


def test_threshold_changes_conclusion(compas):
    """Un umbral más laxo no puede producir MÁS grupos Unfair."""
    strict = mc.run_full_analysis(compas, PROTECTED, "score", "label_value",
                                  "majority", {}, 1.1, "fpr")
    lax = mc.run_full_analysis(compas, PROTECTED, "score", "label_value",
                               "majority", {}, 3.0, "fpr")
    n_unfair_strict = (strict["tables"]["bias_metrics"]["fairness_conclusion"] == "Unfair").sum()
    n_unfair_lax = (lax["tables"]["bias_metrics"]["fairness_conclusion"] == "Unfair").sum()
    assert n_unfair_lax <= n_unfair_strict


def test_fairness_summary_responds_to_tolerance(compas):
    """La conclusión por atributo debe basarse en métricas de error (no ppr) y
    volverse más equitativa al aumentar la tolerancia. Regresión del bug
    'siempre No Equitativo / no responde a la tolerancia'."""
    def summary(tau):
        res = mc.run_full_analysis(compas, PROTECTED, "score", "label_value",
                                   "majority", {}, tau, "fpr")
        fs = res["tables"]["fairness_summary"]
        return dict(zip(fs["attribute_name"], fs["fairness_conclusion"]))

    strict, lax = summary(1.1), summary(3.0)
    # Al menos un atributo debe pasar de Unfair a Fair al relajar la tolerancia.
    flipped = [a for a in strict if strict[a] == "Unfair" and lax[a] == "Fair"]
    assert flipped, f"Ningún atributo respondió a la tolerancia: {strict} -> {lax}"
    # Ningún atributo puede volverse MENOS equitativo al relajar.
    for a in strict:
        assert not (strict[a] == "Fair" and lax[a] == "Unfair")


def test_min_group_size_marks_and_excludes(compas):
    """Subgrupos con muestra insuficiente se marcan y no determinan el veredicto."""
    res = mc.run_full_analysis(compas, PROTECTED, "score", "label_value",
                               "majority", {}, 1.25, "fpr", min_group_size=50)
    bm = res["tables"]["bias_metrics"]
    assert "insufficient_sample" in bm.columns
    # En COMPAS, Asian (n=32) y Native American (n=18) son < 50.
    small = bm[bm["group_size"] < 50]
    assert small["insufficient_sample"].all()
    big = bm[bm["group_size"] >= 50]
    assert not big["insufficient_sample"].any()
    assert res["metadata"]["min_group_size"] == 50


def test_min_group_size_ignores_tiny_group_in_verdict():
    """Un grupo diminuto con disparidad extrema NO debe marcar el atributo Unfair
    si su muestra es insuficiente (pero sí si el umbral es 0)."""
    # Grupo mayoritario H con FPR moderado (0.2) y grupo diminuto M con FPR
    # extremo (1.0) -> disparidad 5x, pero solo 5 personas.
    rows = []
    for i in range(150):  # H negativos: 30 FP, 120 TN -> FPR = 0.2
        rows.append({"sexo": "H", "score": 1 if i < 30 else 0, "label_value": 0})
    for _ in range(150):  # H positivos: todos TP
        rows.append({"sexo": "H", "score": 1, "label_value": 1})
    for _ in range(5):    # M: 5 negativos todos FP -> FPR = 1.0
        rows.append({"sexo": "M", "score": 1, "label_value": 0})
    df = pd.DataFrame(rows)
    strict = mc.run_full_analysis(df, ["sexo"], "score", "label_value",
                                  "majority", {}, 1.25, "fpr", min_group_size=50)
    loose = mc.run_full_analysis(df, ["sexo"], "score", "label_value",
                                 "majority", {}, 1.25, "fpr", min_group_size=0)
    v_strict = strict["tables"]["fairness_summary"].set_index("attribute_name")["fairness_conclusion"]["sexo"]
    v_loose = loose["tables"]["fairness_summary"].set_index("attribute_name")["fairness_conclusion"]["sexo"]
    assert v_strict == "Fair"    # el grupo diminuto se ignora
    assert v_loose == "Unfair"   # sin umbral, sí cuenta


def test_conclusion_based_on_error_metrics(compas):
    """fairness_conclusion = Unfair sii alguna paridad de error (FPR/FNR/FOR/FDR)
    lo es; independiente de la Paridad Estadística (ppr)."""
    res = mc.run_full_analysis(compas, PROTECTED, "score", "label_value",
                               "majority", {}, 1.25, "fpr")
    bm = res["tables"]["bias_metrics"]
    error_parities = ["FPR Parity", "FNR Parity", "FOR Parity", "FDR Parity"]
    for _, row in bm.iterrows():
        expected = "Unfair" if any(row[p] == "Unfair" for p in error_parities) else "Fair"
        assert row["fairness_conclusion"] == expected
