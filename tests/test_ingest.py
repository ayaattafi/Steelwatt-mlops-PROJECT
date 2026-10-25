"""Tests de l'ingestion : le téléchargement réseau est systématiquement simulé."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from steelwatt.data import ingest as ingest_mod


def test_ingest_writes_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, valid_df: pd.DataFrame
) -> None:
    """Le DataFrame téléchargé est écrit tel quel, sans index."""
    monkeypatch.setattr(ingest_mod, "download_dataset", lambda: valid_df)
    out = ingest_mod.ingest(tmp_path / "raw" / "data.csv")
    pd.testing.assert_frame_equal(pd.read_csv(out), valid_df)


def test_ingest_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, valid_df: pd.DataFrame
) -> None:
    """Un fichier déjà présent n'est pas retéléchargé, sauf avec ``force``."""
    calls: list[int] = []

    def fake_download() -> pd.DataFrame:
        """Compte les appels au téléchargement simulé."""
        calls.append(1)
        return valid_df

    monkeypatch.setattr(ingest_mod, "download_dataset", fake_download)
    out = tmp_path / "data.csv"
    ingest_mod.ingest(out)
    ingest_mod.ingest(out)
    assert len(calls) == 1
    ingest_mod.ingest(out, force=True)
    assert len(calls) == 2


def test_download_falls_back_to_zip(
    monkeypatch: pytest.MonkeyPatch, valid_df: pd.DataFrame
) -> None:
    """En cas d'échec de ucimlrepo, l'archive zip officielle est utilisée."""

    def broken() -> pd.DataFrame:
        """Simule une panne de l'API ucimlrepo."""
        raise ConnectionError("API indisponible")

    monkeypatch.setattr(ingest_mod, "fetch_from_ucimlrepo", broken)
    monkeypatch.setattr(ingest_mod, "fetch_from_zip", lambda: valid_df)
    assert ingest_mod.download_dataset() is valid_df
