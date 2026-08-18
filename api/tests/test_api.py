"""Smoke tests de los endpoints FastAPI (sin Aequitas).

Verifican que la respuesta sea JSON estricto (sin token ``NaN``, que rompería
``fetch().json()`` en el navegador) y que todos los endpoints respondan 200.
"""
import io
import json
import os

import pytest
from fastapi.testclient import TestClient

from api.main import app

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CSV_PATH = os.path.join(FIXTURES, "compas_for_aequitas.csv")
COLUMNS = {"protected": ["race", "sex", "age_cat"],
           "predictions": "score", "actual": "label_value"}


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def csv_bytes():
    with open(CSV_PATH, "rb") as f:
        return f.read()


def _assert_strict_json(text):
    """Falla si el cuerpo contiene NaN/Infinity (JSON no estándar)."""
    json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(
        AssertionError(f"Token JSON no estándar: {x}")))


def _full_analysis(client, csv_bytes, method="majority"):
    return client.post(
        "/api/full_analysis",
        files={"file": ("compas.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"columns": json.dumps(COLUMNS),
              "params": json.dumps({"referenceMethod": method,
                                    "fairnessThreshold": 1.25,
                                    "performanceMetric": "fpr"})},
    )


@pytest.mark.parametrize("method", ["majority", "minority", "best_performance"])
def test_full_analysis_ok(client, csv_bytes, method):
    r = _full_analysis(client, csv_bytes, method)
    assert r.status_code == 200
    _assert_strict_json(r.text)
    body = r.json()
    assert set(body["tables"]) >= {
        "group_counts", "group_metrics", "group_metrics_for_plotting",
        "bias_metrics", "fairness_summary"}
    assert body["metadata"]["task_type"] == "binary"
    assert body["plots"]["disparity_summary"].startswith("data:image/png;base64,")


def test_full_analysis_custom_reference(client, csv_bytes):
    r = client.post(
        "/api/full_analysis",
        files={"file": ("compas.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"columns": json.dumps(COLUMNS),
              "params": json.dumps({
                  "referenceMethod": "custom",
                  "referenceGroups": {"race": "Caucasian", "sex": "Male",
                                      "age_cat": "25 - 45"},
                  "fairnessThreshold": 1.25})},
    )
    assert r.status_code == 200
    _assert_strict_json(r.text)


def test_preview(client, csv_bytes):
    r = client.post("/api/preview",
                    files={"file": ("compas.csv", io.BytesIO(csv_bytes), "text/csv")})
    assert r.status_code == 200
    assert set(r.json()) == {"columns", "preview"}


def test_plot_and_recalc_endpoints(client, csv_bytes):
    body = _full_analysis(client, csv_bytes).json()
    bm = body["tables"]["bias_metrics"]
    gm = body["tables"]["group_metrics_for_plotting"]

    r1 = client.post("/api/rerender_plot",
                     json={"bias_metrics": bm, "metrics": ["fpr_disparity"], "attributes": ["race"]})
    assert r1.status_code == 200 and r1.json()["plot"].startswith("data:image")

    r2 = client.post("/api/absolute_plot",
                     json={"group_metrics_for_plotting": gm, "metric": "fpr", "attribute": "race"})
    assert r2.status_code == 200 and r2.json()["plot"].startswith("data:image")

    r3 = client.post("/api/recalculate_fairness",
                     json={"bias_metrics": bm, "fairnessThreshold": 2.0})
    assert r3.status_code == 200
    _assert_strict_json(r3.text)
    assert "fairness_summary" in r3.json()


def test_non_csv_rejected(client):
    r = client.post("/api/preview",
                    files={"file": ("data.txt", io.BytesIO(b"a,b\n1,2"), "text/plain")})
    assert r.status_code == 400
