from pathlib import Path

import pandas as pd
import pytest

from preprocessing.extract import extract
from preprocessing.load import load
from preprocessing.transform import (
    aggregate_daily,
    clean,
    engineer_features,
)


def test_extract_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract([Path("missing.xls")])


def test_clean_removes_duplicates_and_invalid_values():
    df = pd.DataFrame(
        {
            "Date": ["01/01/2024", "01/01/2024", "02/01/2024"],
            "Heures": ["00:00", "00:00", "00:00"],
            "Consommation": [50000, 50000, -10],
            "Prévision J-1": [49000, 49000, 48000],
        }
    )

    result = clean(df)

    assert len(result) == 1
    assert result["Consommation"].iloc[0] == 50000


def test_aggregate_daily():
    df = pd.DataFrame(
        {
            "Date": [
                "01/01/2024",
                "01/01/2024",
                "02/01/2024",
            ],
            "Consommation": [50000, 52000, 51000],
            "Prévision J-1": [49000, 49500, 50000],
        }
    )

    result = aggregate_daily(df)

    assert len(result) == 2
    assert "consommation" in result.columns
    assert "prevision_j1" in result.columns


def test_engineer_features():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")

    df = pd.DataFrame(
        {
            "consommation": [50000 + i for i in range(20)],
            "prevision_j1": [49500 + i for i in range(20)],
        },
        index=dates,
    )

    result = engineer_features(df)

    assert len(result) > 0
    assert "lag_1" in result.columns
    assert "lag_7" in result.columns


def test_load_split():
    dates = pd.date_range("2024-01-01", periods=50, freq="D")

    df = pd.DataFrame(
        {
            "consommation": [50000 + i for i in range(50)],
            "prevision_j1": [49500 + i for i in range(50)],
        },
        index=dates,
    )

    df = engineer_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test, features = load(df)

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    assert len(features) > 0


def test_load_small_dataset():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")

    df = pd.DataFrame(
        {
            "consommation": [50000 + i for i in range(10)],
            "prevision_j1": [49500 + i for i in range(10)],
        },
        index=dates,
    )

    df = engineer_features(df)

    with pytest.raises(ValueError):
        load(df)