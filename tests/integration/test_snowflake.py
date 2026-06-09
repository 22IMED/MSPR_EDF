"""Tests d'intégration — Connexion Snowflake."""

import os
import pytest
import pandas as pd


@pytest.mark.integration
def test_snowflake_connection():
    """Connexion Snowflake établie avec succès."""
    pytest.importorskip("snowflake.connector")
    from pipeline.snowflake_io import _get_connection

    conn = _get_connection()
    assert conn is not None
    conn.close()


@pytest.mark.integration
def test_snowflake_load_data():
    """Chargement des données depuis Snowflake."""
    from pipeline.snowflake_io import load_from_snowflake

    df = load_from_snowflake()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "consommation" in df.columns
    assert "prevision_j1" in df.columns


@pytest.mark.integration
def test_snowflake_data_quality():
    """Les données Snowflake sont de bonne qualité."""
    from pipeline.snowflake_io import load_from_snowflake

    df = load_from_snowflake()
    assert df["consommation"].isna().sum() == 0
    assert (df["consommation"] > 0).all()
    assert (df["consommation"] < 200_000).all()


@pytest.mark.integration
def test_snowflake_date_range():
    """Les données couvrent au moins 2020-2024."""
    from pipeline.snowflake_io import load_from_snowflake

    df = load_from_snowflake()
    assert df.index.min().year <= 2020
    assert df.index.max().year >= 2024
