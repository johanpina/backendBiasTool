"""Tests del soporte multiclase (one-vs-rest) en metrics_core."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from api import metrics_core as mc

PROTECTED = ["genero", "region"]


@pytest.fixture(scope="module")
def multiclass_df():
    rng = np.random.RandomState(7)
    n = 1200
    lab = rng.choice(["A", "B", "C"], n)
    sc = lab.copy()
    flip = rng.rand(n) < 0.4
    sc[flip] = rng.choice(["A", "B", "C"], flip.sum())
    return pd.DataFrame({
        "genero": rng.choice(["M", "F"], n),
        "region": rng.choice(["Norte", "Centro", "Sur"], n),
        "label_value": lab,
        "score": sc,
    })


@pytest.fixture(scope="module")
def binary_df():
    return pd.read_csv(
        __import__("os").path.join(__import__("os").path.dirname(__file__),
                                   "fixtures", "compas_for_aequitas.csv"))


# ---------------------------------------------------------------------------
# Detección de tarea
# ---------------------------------------------------------------------------
def test_detect_task_multiclass(multiclass_df):
    task, classes = mc._detect_task(multiclass_df["label_value"], multiclass_df["score"])
    assert task == "multiclass"
    assert classes == ["A", "B", "C"]


def test_detect_task_binary(binary_df):
    task, classes = mc._detect_task(binary_df["label_value"], binary_df["score"])
    assert task == "binary"


def test_detect_task_two_nonbinary_classes():
    s = pd.Series(["si", "no", "si", "no"])
    task, classes = mc._detect_task(s, s)
    assert task == "multiclass"  # 2 clases no {0,1} -> OvR


# ---------------------------------------------------------------------------
# Estructura de la salida multiclase
# ---------------------------------------------------------------------------
def test_multiclass_structure(multiclass_df):
    res = mc.run_full_analysis(multiclass_df, PROTECTED, "score", "label_value",
                               "majority", {}, 1.25, "fpr")
    assert res["metadata"]["task_type"] == "multiclass"
    assert res["metadata"]["classes"] == ["A", "B", "C"]
    assert set(res["by_class"].keys()) == {"A", "B", "C"}
    # Cada clase expone las tablas en forma binaria (sin columna 'class').
    for cls, entry in res["by_class"].items():
        assert "class" not in entry["tables"]["bias_metrics"].columns
    # Las tablas globales concatenadas SÍ llevan la columna 'class'.
    assert "class" in res["tables"]["bias_metrics"].columns
    assert set(res["tables"]["bias_metrics"]["class"]) == {"A", "B", "C"}


def test_binary_has_no_multiclass_keys(binary_df):
    res = mc.run_full_analysis(binary_df, ["race", "sex"], "score", "label_value",
                               "majority", {}, 1.25, "fpr")
    assert res["metadata"]["task_type"] == "binary"
    assert "by_class" not in res
    assert "class" not in res["tables"]["bias_metrics"].columns


# ---------------------------------------------------------------------------
# Equivalencia OvR: cada clase == correr el binario sobre 'clase vs resto'
# ---------------------------------------------------------------------------
def test_ovr_equals_manual_binarization(multiclass_df):
    res = mc.run_full_analysis(multiclass_df, PROTECTED, "score", "label_value",
                               "majority", {}, 1.25, "fpr")
    for cls in ["A", "B", "C"]:
        manual = multiclass_df.copy()
        manual["score"] = (multiclass_df["score"] == cls).astype(int)
        manual["label_value"] = (multiclass_df["label_value"] == cls).astype(int)
        expected = mc.run_full_analysis(manual, PROTECTED, "score", "label_value",
                                        "majority", {}, 1.25, "fpr")
        # La clase del multiclase debe coincidir con el binario clase-vs-resto.
        assert_frame_equal(
            res["by_class"][cls]["tables"]["group_metrics"].reset_index(drop=True),
            expected["tables"]["group_metrics"].reset_index(drop=True),
        )
        assert_frame_equal(
            res["by_class"][cls]["tables"]["bias_metrics"].reset_index(drop=True),
            expected["tables"]["bias_metrics"].reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# fairness_overall: conservador (Unfair si alguna clase lo es)
# ---------------------------------------------------------------------------
def test_fairness_overall_is_conservative(multiclass_df):
    res = mc.run_full_analysis(multiclass_df, PROTECTED, "score", "label_value",
                               "majority", {}, 1.25, "fpr")
    overall = res["fairness_overall"].set_index("attribute_name")["fairness_conclusion"].to_dict()
    for attr in PROTECTED:
        per_class = [res["by_class"][c]["tables"]["fairness_summary"]
                     .set_index("attribute_name")["fairness_conclusion"].get(attr)
                     for c in ["A", "B", "C"]]
        expected = "Unfair" if "Unfair" in per_class else "Fair"
        assert overall[attr] == expected
