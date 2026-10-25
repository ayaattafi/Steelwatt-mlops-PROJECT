# SteelWatt — Prévision MLOps de la consommation d'une aciérie

[![CI](https://github.com/<organisation>/steelwatt-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/<organisation>/steelwatt-mlops/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![DVC](https://img.shields.io/badge/data-DVC%20%2B%20DagsHub-945dd6)

## Problème métier

La sidérurgie est l'un des secteurs les plus énergivores. La facture d'électricité de
l'aciérie DAEWOO Steel (Gwangyang, Corée du Sud) dépend de l'énergie consommée, des
pics de puissance appelée et de la qualité de l'énergie (puissance réactive, facteur
de puissance).

**SteelWatt prévoit toutes les 15 minutes la consommation de l'heure à venir**
(`Usage_kWh` à *t + 1 h*) à partir de l'historique récent. Si la prévision dépasse un
seuil de pointe, une alerte permet de décaler les opérations non critiques vers les
heures creuses. Objectif : MAE ≤ 5 kWh par intervalle et gain ≥ 20 % par rapport à la
persistance naïve.

On prévoit *t + 1 h* et non l'instant courant, car `Usage_kWh` se déduit presque
exactement de la puissance réactive et du facteur de puissance du même intervalle
(fuite de cible, voir [notebooks/01_eda_check.ipynb](notebooks/01_eda_check.ipynb)).

La fiche de cadrage complète est dans [SYNOPSIS.md](SYNOPSIS.md).

## Architecture

```
UCI Steel Industry ──► DVC + DagsHub (ingest → validate → featurize [lags, t+1h] → split temporel)
                                         │
                                         ▼
   Train (Persistance / Ridge / RF / XGBoost) ──► MLflow Tracking ──► Registry (@champion)
                                         │
                                         ▼
GitHub (PR) ──► Actions CI (lint + pytest) + rapport CML ──► Image Docker ──► Nginx ──► API blue / green
                                         │
                                         ▼
  Flux de production simulé (nov.-déc. + DriftSimulator) ──► Flow Prefect monitoring (KS, χ², PSI)
                                         │                          │
                                         │                          ├──► /metrics ──► Prometheus (alerte)
                                         │                          └──► Rapport HTML Evidently
                                         ▼
               drift_share ≥ 0.30 ──► Flow Prefect de réentraînement (données récentes incluses)
                                         │
               challenger meilleur que le champion (MAE) ? ──► promotion Registry ──► bascule Blue/Green
```

État actuel (**Jalon 1**) : stages DVC `ingest → validate`, tests de validation, CI lint + tests.

## Structure du dépôt

```
steelwatt-mlops/
├── .github/workflows/ci.yml   # CI : flake8, black --check, pytest + couverture
├── data/
│   ├── raw/                   # CSV brut (sortie DVC, non versionné par git)
│   └── processed/             # features (Jalon 2)
├── notebooks/
│   └── 01_eda_check.ipynb     # vérification Load_Type / fuite de cible
├── reports/
│   └── validation.json        # métriques DVC de l'étape validate
├── src/steelwatt/
│   ├── data/                  # ingest.py, validate.py
│   ├── features/              # Jalon 2
│   ├── models/                # Jalon 2
│   ├── api/                   # Jalon 3
│   ├── monitoring/            # Jalon 4
│   └── pipelines/             # Jalon 4 (Prefect)
├── tests/                     # tests pytest (sans réseau)
├── dvc.yaml / dvc.lock        # pipeline DVC
├── pyproject.toml             # packaging + black / pytest / coverage
├── requirements.txt           # dépendances épinglées
├── requirements-dev.txt       # outils de dev (tests, lint, notebook)
└── SYNOPSIS.md                # fiche de cadrage
```

## Installation et reproduction

Prérequis : Python 3.11.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt && pip install -e . --no-deps
```

Reproduction en 3 commandes :

```bash
git clone https://github.com/<organisation>/steelwatt-mlops.git && cd steelwatt-mlops
dvc pull        # récupère data/raw depuis le remote DagsHub
pytest --cov    # tests + couverture
```

Sans accès au remote, `dvc repro` retélécharge le dataset depuis UCI puis le valide.

## Équipe

| Membre | Rôle |
|---|---|
| ATTAFI Aya | Lead Data & DVC + Modeling & MLflow |
| CHERNI Louay | Lead Serving & CI/CD + Déploiement Blue/Green |
| KHADHRAOUI Tayf | Lead Observabilité & SRE + Orchestration Prefect |

## Dataset et citation

*Steel Industry Energy Consumption*, UCI Machine Learning Repository (id 851), 2021.
DOI : [10.24432/C52G8C](https://doi.org/10.24432/C52G8C) — licence
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

```bibtex
@misc{steel_industry_energy_consumption_851,
  author       = {E, Sathishkumar V and Cho, Yongyun},
  title        = {{Steel Industry Energy Consumption}},
  year         = {2021},
  howpublished = {UCI Machine Learning Repository},
  note         = {DOI: https://doi.org/10.24432/C52G8C}
}
```
