"""Validation du dataset brut (étape DVC ``validate``).

Contrôles effectués :

* schéma : 11 colonnes aux noms exacts, types attendus, nombre de lignes ;
* complétude : aucune valeur manquante ;
* bornes : puissances ≥ 0, facteurs de puissance dans [0, 100], NSM dans [0, 86400[ ;
* catégories autorisées pour ``WeekStatus``, ``Day_of_week`` et ``Load_Type`` ;
* temporalité : pas régulier de 15 min, sans doublon, cohérence ``NSM`` / ``date``.

Un rapport JSON est écrit dans tous les cas, puis une ``DataValidationError``
est levée si au moins un contrôle échoue.

Usage ::

    python -m steelwatt.data.validate [--input ...] [--report reports/validation.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_INPUT: Path = Path("data/raw/steel_industry_data.csv")
DEFAULT_REPORT: Path = Path("reports/validation.json")

EXPECTED_ROWS: int = 35_040
DATE_FORMAT: str = "%d/%m/%Y %H:%M"
STEP: pd.Timedelta = pd.Timedelta(minutes=15)
SECONDS_PER_DAY: int = 86_400

# Nom exact de colonne -> type logique attendu.
EXPECTED_SCHEMA: dict[str, str] = {
    "date": "datetime",
    "Usage_kWh": "float",
    "Lagging_Current_Reactive.Power_kVarh": "float",
    "Leading_Current_Reactive_Power_kVarh": "float",
    "CO2(tCO2)": "float",
    "Lagging_Current_Power_Factor": "float",
    "Leading_Current_Power_Factor": "float",
    "NSM": "int",
    "WeekStatus": "category",
    "Day_of_week": "category",
    "Load_Type": "category",
}

NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
)
POWER_FACTOR_COLUMNS: tuple[str, ...] = (
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
)
ALLOWED_CATEGORIES: dict[str, frozenset[str]] = {
    "WeekStatus": frozenset({"Weekday", "Weekend"}),
    "Day_of_week": frozenset(
        {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    ),
    "Load_Type": frozenset({"Light_Load", "Medium_Load", "Maximum_Load"}),
}


class DataValidationError(Exception):
    """Levée lorsqu'au moins un contrôle de validation échoue."""


@dataclass
class CheckResult:
    """Résultat d'un contrôle de validation.

    Attributes:
        name: Identifiant du contrôle.
        passed: ``True`` si le contrôle est satisfait.
        details: Informations de diagnostic (valeurs observées, exemples).
    """

    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


def _matches_kind(series: pd.Series, kind: str) -> bool:
    """Indique si une série correspond au type logique attendu.

    Args:
        series: Colonne à tester.
        kind: Type logique (``datetime``, ``float``, ``int`` ou ``category``).

    Returns:
        ``True`` si le dtype est compatible.
    """
    if kind == "float":
        return pd.api.types.is_numeric_dtype(series) and not (
            pd.api.types.is_bool_dtype(series)
        )
    if kind == "int":
        return pd.api.types.is_integer_dtype(series)
    if kind == "category":
        return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(
            series
        )
    if kind == "datetime":
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        parsed = pd.to_datetime(series, format=DATE_FORMAT, errors="coerce")
        return bool(parsed.notna().all())
    raise ValueError(f"Type logique inconnu : {kind}")


def check_columns(df: pd.DataFrame) -> CheckResult:
    """Vérifie la présence exacte des 11 colonnes attendues.

    Args:
        df: Dataset brut.

    Returns:
        Le résultat du contrôle (colonnes manquantes / inattendues).
    """
    expected = list(EXPECTED_SCHEMA)
    missing = [c for c in expected if c not in df.columns]
    unexpected = [c for c in df.columns if c not in EXPECTED_SCHEMA]
    return CheckResult(
        "columns",
        not missing and not unexpected,
        {"n_columns": int(df.shape[1]), "missing": missing, "unexpected": unexpected},
    )


def check_dtypes(df: pd.DataFrame) -> CheckResult:
    """Vérifie le type de chaque colonne attendue présente.

    Args:
        df: Dataset brut.

    Returns:
        Le résultat du contrôle (colonnes au mauvais type).
    """
    wrong = {
        col: str(df[col].dtype)
        for col, kind in EXPECTED_SCHEMA.items()
        if col in df.columns and not _matches_kind(df[col], kind)
    }
    return CheckResult("dtypes", not wrong, {"wrong_dtypes": wrong})


def check_row_count(df: pd.DataFrame, expected_rows: int | None) -> CheckResult:
    """Vérifie le nombre de lignes.

    Args:
        df: Dataset brut.
        expected_rows: Nombre attendu (``None`` désactive le contrôle).

    Returns:
        Le résultat du contrôle.
    """
    passed = expected_rows is None or len(df) == expected_rows
    return CheckResult(
        "row_count", passed, {"n_rows": int(len(df)), "expected": expected_rows}
    )


def check_missing_values(df: pd.DataFrame) -> CheckResult:
    """Vérifie l'absence de valeurs manquantes.

    Args:
        df: Dataset brut.

    Returns:
        Le résultat du contrôle (nombre de NaN par colonne fautive).
    """
    counts = df.isna().sum()
    missing = {col: int(n) for col, n in counts.items() if n > 0}
    return CheckResult("missing_values", not missing, {"missing": missing})


def check_bounds(df: pd.DataFrame) -> CheckResult:
    """Vérifie les bornes physiques des grandeurs numériques.

    Args:
        df: Dataset brut.

    Returns:
        Le résultat du contrôle (nombre de valeurs hors bornes par colonne).
    """
    rules: dict[str, tuple[float, float, bool]] = {
        **{c: (0.0, float("inf"), True) for c in NON_NEGATIVE_COLUMNS},
        **{c: (0.0, 100.0, True) for c in POWER_FACTOR_COLUMNS},
        "NSM": (0.0, float(SECONDS_PER_DAY), False),  # borne haute exclue
    }
    violations: dict[str, int] = {}
    for col, (low, high, high_inclusive) in rules.items():
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = df[col]
        above = values > high if high_inclusive else values >= high
        n_bad = int(((values < low) | above).sum())
        if n_bad:
            violations[col] = n_bad
    return CheckResult("bounds", not violations, {"out_of_bounds": violations})


def check_categories(df: pd.DataFrame) -> CheckResult:
    """Vérifie que les variables catégorielles n'ont que des modalités autorisées.

    Args:
        df: Dataset brut.

    Returns:
        Le résultat du contrôle (modalités inconnues par colonne).
    """
    unknown: dict[str, list[str]] = {}
    for col, allowed in ALLOWED_CATEGORIES.items():
        if col not in df.columns:
            continue
        observed = set(df[col].dropna().astype(str).unique())
        extra = sorted(observed - allowed)
        if extra:
            unknown[col] = extra
    return CheckResult("categories", not unknown, {"unknown": unknown})


def parse_timestamps(dates: pd.Series) -> pd.Series:
    """Convertit la colonne ``date`` brute en horodatages de fin d'intervalle.

    Dans le CSV UCI, l'intervalle 23:45-24:00 du jour J est daté « J 00:00 »
    (et non « J+1 00:00 ») : la ligne de minuit porte la date du jour qu'elle
    clôture. On décale donc ces lignes d'un jour pour obtenir une série
    strictement croissante au pas de 15 min.

    Args:
        dates: Colonne ``date`` au format ``JJ/MM/AAAA HH:MM``.

    Returns:
        Les horodatages corrigés (``NaT`` si une date est illisible).
    """
    parsed = pd.to_datetime(dates, format=DATE_FORMAT, errors="coerce")
    is_midnight = parsed.dt.hour.eq(0) & parsed.dt.minute.eq(0)
    return parsed + pd.to_timedelta(is_midnight.astype(int), unit="D")


def check_time_step(df: pd.DataFrame) -> CheckResult:
    """Vérifie la régularité du pas de 15 min et la cohérence NSM / date.

    Args:
        df: Dataset brut.

    Returns:
        Le résultat du contrôle (irrégularités, doublons, incohérences NSM).
    """
    if "date" not in df.columns:
        return CheckResult("time_step", False, {"error": "colonne date absente"})
    dates = parse_timestamps(df["date"])
    if dates.isna().any():
        return CheckResult(
            "time_step", False, {"unparseable_dates": int(dates.isna().sum())}
        )
    diffs = dates.diff().dropna()
    irregular = diffs[diffs != STEP]
    details: dict[str, Any] = {
        "start": str(dates.iloc[0]),
        "end": str(dates.iloc[-1]),
        "n_irregular_steps": int(len(irregular)),
        "n_duplicates": int(dates.duplicated().sum()),
        "n_midnight_rollovers": int(
            (dates.dt.hour.eq(0) & dates.dt.minute.eq(0)).sum()
        ),
        "examples": [str(dates.iloc[i]) for i in irregular.index[:5]],
    }
    passed = details["n_irregular_steps"] == 0 and details["n_duplicates"] == 0
    if "NSM" in df.columns and pd.api.types.is_numeric_dtype(df["NSM"]):
        nsm_from_date = (dates - dates.dt.normalize()).dt.total_seconds()
        n_mismatch = int((nsm_from_date != df["NSM"]).sum())
        details["n_nsm_mismatch"] = n_mismatch
        passed = passed and n_mismatch == 0
    return CheckResult("time_step", passed, details)


def validate_dataframe(
    df: pd.DataFrame, expected_rows: int | None = EXPECTED_ROWS
) -> dict[str, Any]:
    """Exécute l'ensemble des contrôles sur un DataFrame.

    Args:
        df: Dataset brut.
        expected_rows: Nombre de lignes attendu (``None`` pour ne pas contrôler).

    Returns:
        Un rapport sérialisable en JSON avec le statut global et le détail.
    """
    checks = [
        check_columns(df),
        check_dtypes(df),
        check_row_count(df, expected_rows),
        check_missing_values(df),
        check_bounds(df),
        check_categories(df),
        check_time_step(df),
    ]
    n_failed = sum(not c.passed for c in checks)
    return {
        "passed": n_failed == 0,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_checks": len(checks),
        "n_checks_failed": n_failed,
        "checks": {c.name: asdict(c) for c in checks},
    }


def write_report(report: dict[str, Any], report_path: Path) -> None:
    """Écrit le rapport de validation au format JSON.

    Args:
        report: Rapport produit par :func:`validate_dataframe`.
        report_path: Chemin du fichier JSON.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def validate_file(
    input_path: Path = DEFAULT_INPUT,
    report_path: Path = DEFAULT_REPORT,
    expected_rows: int | None = EXPECTED_ROWS,
) -> dict[str, Any]:
    """Valide un CSV brut, écrit le rapport et lève une erreur en cas d'échec.

    Args:
        input_path: CSV à valider.
        report_path: Chemin du rapport JSON.
        expected_rows: Nombre de lignes attendu.

    Returns:
        Le rapport de validation.

    Raises:
        DataValidationError: Si au moins un contrôle échoue.
    """
    df = pd.read_csv(input_path)
    report = validate_dataframe(df, expected_rows=expected_rows)
    write_report(report, report_path)
    for name, check in report["checks"].items():
        level = logging.INFO if check["passed"] else logging.ERROR
        logger.log(level, "%-15s %s %s", name, check["passed"], check["details"])
    if not report["passed"]:
        failed = [n for n, c in report["checks"].items() if not c["passed"]]
        raise DataValidationError(f"Contrôles en échec : {', '.join(failed)}")
    logger.info(
        "Validation réussie (%d contrôles) -> %s", report["n_checks"], report_path
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande.

    Args:
        argv: Liste d'arguments (``None`` = ``sys.argv``).

    Returns:
        Les arguments analysés.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Point d'entrée CLI de la validation.

    Args:
        argv: Liste d'arguments (``None`` = ``sys.argv``).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    args = parse_args(argv)
    validate_file(args.input, args.report)


if __name__ == "__main__":
    main()
