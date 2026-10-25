"""Téléchargement du dataset UCI « Steel Industry Energy Consumption » (id 851).

Le module est idempotent : si le CSV brut existe déjà et n'est pas vide, aucun
téléchargement n'est relancé (sauf avec ``--force``). La source principale est
``ucimlrepo`` ; en cas d'échec, l'archive zip officielle d'UCI est utilisée.

Usage ::

    python -m steelwatt.data.ingest [--output data/raw/steel_industry_data.csv]
"""

from __future__ import annotations

import argparse
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

UCI_DATASET_ID: int = 851
UCI_ZIP_URL: str = (
    "https://archive.ics.uci.edu/static/public/851/"
    "steel+industry+energy+consumption.zip"
)
DEFAULT_OUTPUT: Path = Path("data/raw/steel_industry_data.csv")
DOWNLOAD_TIMEOUT_S: int = 60


def fetch_from_ucimlrepo(dataset_id: int = UCI_DATASET_ID) -> pd.DataFrame:
    """Télécharge le dataset via le client officiel ``ucimlrepo``.

    Args:
        dataset_id: Identifiant UCI du dataset.

    Returns:
        Le DataFrame original (toutes les colonnes, y compris ``date``).
    """
    from ucimlrepo import fetch_ucirepo

    logger.info("Téléchargement via ucimlrepo (id=%d)", dataset_id)
    repo = fetch_ucirepo(id=dataset_id)
    return repo.data.original.copy()


def fetch_from_zip(url: str = UCI_ZIP_URL) -> pd.DataFrame:
    """Télécharge l'archive zip officielle et lit le premier CSV qu'elle contient.

    Args:
        url: URL de l'archive zip UCI.

    Returns:
        Le DataFrame lu depuis le CSV de l'archive.

    Raises:
        FileNotFoundError: Si l'archive ne contient aucun fichier CSV.
    """
    logger.info("Téléchargement de l'archive officielle : %s", url)
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"Aucun CSV dans l'archive {url}")
        with archive.open(csv_names[0]) as handle:
            return pd.read_csv(handle)


def download_dataset() -> pd.DataFrame:
    """Récupère le dataset en essayant ``ucimlrepo`` puis l'archive zip.

    Returns:
        Le DataFrame brut du dataset.
    """
    try:
        return fetch_from_ucimlrepo()
    except Exception as exc:  # noqa: BLE001 - toute erreur réseau/API → repli
        logger.warning("Échec ucimlrepo (%s), repli sur l'archive zip", exc)
        return fetch_from_zip()


def ingest(output_path: Path = DEFAULT_OUTPUT, force: bool = False) -> Path:
    """Écrit le CSV brut sur disque, sans retélécharger s'il existe déjà.

    Args:
        output_path: Chemin du CSV de sortie.
        force: Si ``True``, retélécharge même si le fichier existe.

    Returns:
        Le chemin du CSV écrit (ou déjà présent).
    """
    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        logger.info("%s existe déjà, téléchargement ignoré", output_path)
        return output_path

    df = download_dataset()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)  # écriture atomique
    logger.info(
        "Dataset écrit : %s (%d lignes, %d colonnes)",
        output_path,
        len(df),
        df.shape[1],
    )
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande.

    Args:
        argv: Liste d'arguments (``None`` = ``sys.argv``).

    Returns:
        Les arguments analysés.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="retélécharger")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Point d'entrée CLI de l'ingestion.

    Args:
        argv: Liste d'arguments (``None`` = ``sys.argv``).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    args = parse_args(argv)
    ingest(args.output, force=args.force)


if __name__ == "__main__":
    main()
