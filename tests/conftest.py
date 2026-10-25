"""Fixtures partagées : petits DataFrames synthétiques au format du dataset UCI."""

from __future__ import annotations

import pandas as pd
import pytest

from steelwatt.data.validate import DATE_FORMAT


def make_synthetic_df(n_rows: int = 8) -> pd.DataFrame:
    """Construit un DataFrame valide qui imite le schéma du dataset brut.

    Args:
        n_rows: Nombre d'intervalles de 15 min à générer.

    Returns:
        Un DataFrame conforme à tous les contrôles de validation.
    """
    dates = pd.date_range("2018-01-01 00:15", periods=n_rows, freq="15min")
    nsm = (dates - dates.normalize()).total_seconds().astype(int)
    # Convention UCI : la ligne de minuit porte la date du jour qu'elle clôture.
    labels = (dates - pd.to_timedelta((nsm == 0).astype(int), unit="D")).strftime(
        DATE_FORMAT
    )
    return pd.DataFrame(
        {
            "date": labels,
            "Usage_kWh": [3.17 + i for i in range(n_rows)],
            "Lagging_Current_Reactive.Power_kVarh": [2.95] * n_rows,
            "Leading_Current_Reactive_Power_kVarh": [0.0] * n_rows,
            "CO2(tCO2)": [0.0] * n_rows,
            "Lagging_Current_Power_Factor": [73.21] * n_rows,
            "Leading_Current_Power_Factor": [100.0] * n_rows,
            "NSM": nsm,
            "WeekStatus": ["Weekday"] * n_rows,
            "Day_of_week": ["Monday"] * n_rows,
            "Load_Type": ["Light_Load"] * n_rows,
        }
    )


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """Fixture : DataFrame synthétique valide de 8 lignes.

    Returns:
        Le DataFrame synthétique.
    """
    return make_synthetic_df()
