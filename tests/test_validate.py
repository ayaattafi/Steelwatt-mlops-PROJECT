"""Tests du module de validation sur DataFrames synthétiques (sans réseau)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from steelwatt.data.validate import (
    DataValidationError,
    parse_timestamps,
    validate_dataframe,
    validate_file,
)
from tests.conftest import make_synthetic_df


def _failed(report: dict) -> set[str]:
    """Renvoie les noms des contrôles en échec.

    Args:
        report: Rapport de validation.

    Returns:
        L'ensemble des contrôles échoués.
    """
    return {name for name, c in report["checks"].items() if not c["passed"]}


def test_valid_dataframe_passes(valid_df: pd.DataFrame) -> None:
    """Un DataFrame conforme passe tous les contrôles."""
    report = validate_dataframe(valid_df, expected_rows=len(valid_df))
    assert report["passed"], _failed(report)
    assert report["n_checks_failed"] == 0


def test_missing_column_fails(valid_df: pd.DataFrame) -> None:
    """Une colonne manquante fait échouer le contrôle de schéma."""
    report = validate_dataframe(valid_df.drop(columns="Load_Type"), expected_rows=8)
    assert not report["passed"]
    assert "columns" in _failed(report)
    assert report["checks"]["columns"]["details"]["missing"] == ["Load_Type"]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Usage_kWh", -1.0),
        ("Lagging_Current_Reactive.Power_kVarh", -0.5),
        ("Lagging_Current_Power_Factor", 100.5),
        ("Leading_Current_Power_Factor", -3.0),
    ],
)
def test_out_of_bounds_value_fails(
    valid_df: pd.DataFrame, column: str, value: float
) -> None:
    """Une valeur hors bornes physiques fait échouer le contrôle de bornes."""
    valid_df.loc[2, column] = value
    report = validate_dataframe(valid_df, expected_rows=8)
    assert _failed(report) == {"bounds"}
    assert report["checks"]["bounds"]["details"]["out_of_bounds"] == {column: 1}


def test_nsm_upper_bound_is_exclusive(valid_df: pd.DataFrame) -> None:
    """NSM = 86400 est hors de l'intervalle [0, 86400[."""
    valid_df.loc[0, "NSM"] = 86_400
    report = validate_dataframe(valid_df, expected_rows=8)
    assert "bounds" in _failed(report)


def test_unknown_category_fails(valid_df: pd.DataFrame) -> None:
    """Une modalité inconnue fait échouer le contrôle de catégories."""
    valid_df.loc[3, "Load_Type"] = "Overload"
    report = validate_dataframe(valid_df, expected_rows=8)
    assert _failed(report) == {"categories"}
    assert report["checks"]["categories"]["details"]["unknown"] == {
        "Load_Type": ["Overload"]
    }


def test_missing_value_fails(valid_df: pd.DataFrame) -> None:
    """Une valeur manquante fait échouer le contrôle de complétude."""
    valid_df.loc[1, "Usage_kWh"] = None
    report = validate_dataframe(valid_df, expected_rows=8)
    assert "missing_values" in _failed(report)


def test_wrong_row_count_fails(valid_df: pd.DataFrame) -> None:
    """Un nombre de lignes différent de l'attendu est détecté."""
    report = validate_dataframe(valid_df)  # attendu par défaut : 35 040
    assert _failed(report) == {"row_count"}


def test_wrong_dtype_fails(valid_df: pd.DataFrame) -> None:
    """Une colonne numérique stockée en texte est détectée."""
    valid_df["NSM"] = valid_df["NSM"].astype(str)
    report = validate_dataframe(valid_df, expected_rows=8)
    assert "dtypes" in _failed(report)


def test_irregular_time_step_fails(valid_df: pd.DataFrame) -> None:
    """Un trou dans la série temporelle est détecté."""
    report = validate_dataframe(valid_df.drop(index=4), expected_rows=7)
    assert _failed(report) == {"time_step"}
    assert report["checks"]["time_step"]["details"]["n_irregular_steps"] == 1


def test_midnight_rollover_convention() -> None:
    """La ligne « J 00:00 » qui suit « J 23:45 » est interprétée comme J+1 00:00."""
    df = make_synthetic_df(n_rows=100)  # franchit minuit
    assert df["date"].iloc[95] == "01/01/2018 00:00"
    stamps = parse_timestamps(df["date"])
    assert stamps.iloc[95] == pd.Timestamp("2018-01-02 00:00")
    report = validate_dataframe(df, expected_rows=100)
    assert report["passed"], _failed(report)
    assert report["checks"]["time_step"]["details"]["n_midnight_rollovers"] == 1


def test_validate_file_writes_report_and_raises(
    valid_df: pd.DataFrame, tmp_path: Path
) -> None:
    """``validate_file`` écrit le rapport puis lève une erreur en cas d'échec."""
    csv_path = tmp_path / "raw.csv"
    report_path = tmp_path / "reports" / "validation.json"
    valid_df.to_csv(csv_path, index=False)

    report = validate_file(csv_path, report_path, expected_rows=8)
    assert report["passed"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True

    valid_df.loc[0, "WeekStatus"] = "Holiday"
    valid_df.to_csv(csv_path, index=False)
    with pytest.raises(DataValidationError, match="categories"):
        validate_file(csv_path, report_path, expected_rows=8)
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is False
