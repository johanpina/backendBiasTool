"""Tests del módulo de análisis exploratorio (EDA)."""
import io
import json
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.eda import run_eda, cramers_v
from api.main import app

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="module")
def compas():
    return pd.read_csv(os.path.join(FIXTURES, "compas_for_aequitas.csv"))


def test_eda_structure(compas):
    r = run_eda(compas)
    assert r["n_rows"] == len(compas)
    assert r["n_cols"] == compas.shape[1]
    assert {c["name"] for c in r["columns"]} == set(compas.columns)
    assert "matrix" in r["associations"] and "columns" in r["associations"]


def test_eda_dtype_detection(compas):
    r = run_eda(compas)
    by_name = {c["name"]: c for c in r["columns"]}
    assert by_name["entity_id"]["dtype"] == "numeric"
    assert by_name["score"]["dtype"] == "binary"
    assert by_name["race"]["dtype"] == "categorical"
    # entity_id no debe entrar en la matriz de asociación (alta cardinalidad).
    assert "entity_id" not in r["associations"]["columns"]


def test_eda_small_group_and_cardinality_alerts(compas):
    r = run_eda(compas, min_group_size=50)
    types = {a["type"] for a in r["alerts"]}
    assert "small_group" in types      # Asian/Native American en race
    assert "high_cardinality" in types  # entity_id


def test_eda_detects_proxy():
    """Dos columnas casi idénticas deben marcarse como posible proxy."""
    rng = np.random.RandomState(0)
    comuna = rng.choice(["A", "B", "C", "D"], 600)
    # nivel socioeconómico determinado casi 1:1 por la comuna
    nse = np.where(comuna == "A", "Alto",
          np.where(comuna == "B", "Medio",
          np.where(comuna == "C", "Bajo", "Bajo")))
    df = pd.DataFrame({"comuna": comuna, "nse": nse,
                       "label_value": rng.randint(0, 2, 600),
                       "score": rng.randint(0, 2, 600)})
    r = run_eda(df)
    proxy = [a for a in r["alerts"] if a["type"] == "proxy"]
    assert any(set(a["columns"]) == {"comuna", "nse"} for a in proxy)


def test_cramers_v_bounds():
    x = pd.Series(["a", "a", "b", "b"])
    assert cramers_v(x, x) == pytest.approx(1.0, abs=1e-9)
    indep = pd.Series(["a", "b", "a", "b"])
    assert 0.0 <= cramers_v(x, indep) <= 1.0


def test_eda_enrichment(compas):
    """Histograma para numéricas, balance/role para categóricas."""
    r = run_eda(compas)
    by_name = {c["name"]: c for c in r["columns"]}
    # entity_id numérica -> histograma
    assert by_name["entity_id"]["histogram"]
    assert all({"x0", "x1", "count"} <= set(b) for b in by_name["entity_id"]["histogram"])
    # race categórica -> balance + role protegida
    assert 0.0 <= by_name["race"]["balance"]["evenness"] <= 1.0
    assert by_name["race"]["role_hint"] == "protected"
    # sexo binario de texto -> protegida (no outcome)
    assert by_name["sex"]["role_hint"] == "protected"
    # score numérico 0/1 -> outcome
    assert by_name["score"]["role_hint"] == "outcome"


def test_eda_crosstabs(compas):
    """Tablas de contingencia precomputadas y coherentes con los conteos."""
    r = run_eda(compas)
    assert "race" in r["crosstab_columns"] and "score" in r["crosstab_columns"]
    ct = r["crosstabs"]["race|||score"]
    assert len(ct["counts"]) == len(ct["x_values"])
    assert len(ct["counts"][0]) == len(ct["y_values"])
    # El total de la tabla no supera el nº de filas.
    total = sum(sum(row) for row in ct["counts"])
    assert total <= r["n_rows"]
    # Coincide con un value_counts manual para una celda.
    i = ct["x_values"].index("African-American")
    j = ct["y_values"].index("1.0")
    manual = int(((compas["race"] == "African-American") & (compas["score"] == 1)).sum())
    assert ct["counts"][i][j] == manual


def test_eda_endpoint(compas):
    client = TestClient(app)
    with open(os.path.join(FIXTURES, "compas_for_aequitas.csv"), "rb") as f:
        r = client.post("/api/eda", files={"file": ("c.csv", f, "text/csv")})
    assert r.status_code == 200
    # JSON estricto (sin NaN)
    json.loads(r.text, parse_constant=lambda x: (_ for _ in ()).throw(
        AssertionError(f"Token no estándar: {x}")))
    body = r.json()
    assert body["n_rows"] == len(compas)
    assert isinstance(body["alerts"], list)
