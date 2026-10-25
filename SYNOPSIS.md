# FICHE DE CADRAGE : SYNOPSIS DU PROJET MLOPS (SESSION 2026–2027)

---

## 1. Informations Générales & Équipe

* **Titre du Projet :** **SteelWatt** — Plateforme MLOps de Prévision de la Consommation Énergétique d'une Aciérie
* **Taille du groupe :** 3 étudiants (Trinôme)
* **Membres de l'équipe :**
  1. **Étudiant 1 (Lead Data & DVC + Modeling & MLflow) :** ATTAFI Aya — Email : `...`
  2. **Étudiant 2 (Lead Serving & CI/CD + Déploiement Blue/Green) :** CHERNI Louay — Email : `...`
  3. **Étudiant 3 (Lead Observabilité & SRE + Orchestration Prefect) :** KHADHRAOUI Tayf — Email : `...`
* **Lien vers le Dépôt GitHub :** `https://github.com/<organisation>/steelwatt-mlops`

---

## 2. Problématique Métier & Cas d'Usage

* **Contexte Business :** Industrie sidérurgique — l'acier est l'un des secteurs les plus énergivores. L'usine étudiée (DAEWOO Steel, Gwangyang, Corée du Sud) produit bobines, tôles et plaques d'acier. Sa facture d'électricité dépend de la quantité consommée, des pics de puissance appelée et de la qualité de l'énergie (puissance réactive, facteur de puissance). Sans anticipation, l'usine subit des pics coûteux, des dépassements de puissance souscrite et des émissions de CO2 évitables.
* **Décision assistée ou automatisée :** Toutes les 15 minutes, le modèle **prévoit la consommation de l'heure à venir** (kWh à t + 1 h) à partir de l'historique récent des grandeurs électriques et du contexte de production. Si la consommation prévue dépasse un seuil de pointe, le système **émet une alerte** au responsable énergie / production pour décaler les opérations non critiques vers les heures creuses et vérifier la compensation de l'énergie réactive.
* **Objectif métier quantifié (KPI) :** Prévoir la consommation à 1 h avec une **erreur moyenne ≤ 5 kWh par intervalle** et anticiper les intervalles de forte charge afin de lisser les pics de puissance et réduire la facture énergétique.

---

## 3. Formulation Machine Learning & Choix du Dataset

* **Type de problème ML :** Régression continue (série temporelle reformulée en régression supervisée, horizon 1 h = 4 intervalles)
* **Variable Cible (`Target`) :** `Usage_kWh_t+1h` — énergie électrique consommée par l'usine sur l'intervalle de 15 minutes situé une heure après l'instant de prévision (kWh)
* **Source du Dataset :** Steel Industry Energy Consumption — UCI ML Repository : `https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption` — DOI `10.24432/C52G8C` — Licence **CC BY 4.0**
* **Volume et dimensions :** 35 040 lignes (une mesure toutes les 15 min sur l'année 2018), 11 colonnes brutes, 2,6 Mo, aucune valeur manquante
* **Variables prédictives clés** (uniquement des valeurs connues à l'instant de prévision) :
  * *Variables numériques :* `Usage_kWh` actuel et retardé (`lag_1`, `lag_4`, `lag_96` = même heure la veille), moyenne glissante sur 1 h, `Lagging_Current_Reactive.Power_kVarh`, `Leading_Current_Reactive_Power_kVarh`, `Lagging_Current_Power_Factor`, `Leading_Current_Power_Factor` (valeurs de l'intervalle courant), `NSM` (secondes depuis minuit)
  * *Variables catégorielles :* `WeekStatus` (semaine / week-end), `Day_of_week` (lundi … dimanche), `Load_Type` (Light / Medium / Maximum Load)
  * *Variables exclues :* `CO2(tCO2)`, proportionnelle à la consommation ; `date`, utilisée uniquement pour le découpage et les features calendaires.
  * *Pourquoi une prévision et non une estimation :* la puissance active (kWh) se déduit presque exactement de la puissance réactive et du facteur de puissance **du même intervalle** ($P = Q \cdot \frac{\cos\varphi}{\sqrt{1-\cos^2\varphi}}$). Prédire `Usage_kWh` au même instant serait une fuite de cible ; on prédit donc l'heure suivante à partir des mesures passées.
* **Découpage temporel :** entraînement janvier – août, validation septembre – octobre, flux de « production » novembre – décembre (pas de mélange aléatoire ; les lags sont calculés avant le découpage).
* **Modèles comparés (MLflow) :** Persistance naïve (consommation actuelle = prévision, référence métier), Régression Ridge, Random Forest, XGBoost.
* **Métriques ML cibles :** MAE $\le 5$ kWh, RMSE $\le 8$ kWh, $R^2 \ge 0.85$, gain $\ge 20\%$ de MAE par rapport à la persistance naïve, Latence d'inférence $< 100$ ms (seuils à recalibrer après la baseline du Jalon 2 ; la MAPE est écartée car la consommation est proche de zéro en charge faible).

---

## 4. Architecture MLOps Prévisionnelle

| Composant MLOps | Outil & Configuration retenue |
|---|---|
| **Versioning Données** | DVC avec remote storage cloud **DagsHub** (accessible depuis GitHub Actions, reproduction par `git clone` → `dvc pull` → `pytest`) ; étapes `dvc.yaml` : `ingest → validate → featurize → split → train → evaluate` |
| **Tracking & Registry** | MLflow Tracking Server local (`http://127.0.0.1:5000`) & Model Registry : comparaison des 4 modèles (MAE, RMSE, R², courbe réel vs prévu), promotion du champion en Production (alias `@champion`) |
| **Microservice Inférence** | API REST FastAPI (`/predict` unitaire et batch, `/health`, `/metrics`) avec documentation Swagger automatique |
| **Validation des données** | Schémas Pydantic stricts pour les payloads d'entrée (historique des 96 derniers intervalles, puissances réactives ≥ 0, facteurs de puissance dans [0 ; 100] %, `NSM` dans [0 ; 86 400[, catégories autorisées) ; validation du dataset brut à l'étape DVC `validate` |
| **Packaging & Isolation** | Image Docker multi-stage `python:3.11-slim` non-root & `docker-compose.yml` (API blue/green, Nginx, MLflow, Prometheus) |
| **Intégration Continue (CI)** | GitHub Actions : Linting (`flake8`, `black`) + Pytest (unitaires, schéma, features sans fuite temporelle, inférence), couverture > 75 % |
| **Rapports CML** | Publication automatique en commentaire de PR : métriques MAE / RMSE / R² vs baseline et graphe réel vs prévu |
| **Orchestration** | Prefect 3.x (Tasks atomiques avec retries, Flow d'entraînement, Flow de monitoring, Work Pool `process` + CRON) |
| **Déploiement Continu** | Stratégie Zero-Downtime **Blue/Green** via `docker-compose` : deux conteneurs API (`api-blue`, `api-green`) derrière un reverse-proxy Nginx, bascule du trafic après validation du `/health` du nouveau modèle |
| **Surveillance de Dérive** | Tests statistiques Kolmogorov-Smirnov (numériques), $\chi^2$ (catégorielles) et calcul formel du PSI + dashboard interactif d'observabilité HTML (Evidently) |
| **Export Métriques** | Format OpenMetrics Prometheus (`/metrics`) scrapé par un serveur Prometheus : PSI par feature, `drift_share`, MAE glissante, latence, nombre de prédictions ; règle d'alerte sur `drift_share` |
| **Boucle de Rétroaction** | Déclenchement automatique du réentraînement si `drift_share >= 0.30` |

**Pipeline MLOps de bout en bout :**

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

---

## 5. Scénario de Dérive de Données (Data Drift) Simulé

* **Quelles variables vont dériver en production ?**
  * **Vieillissement des batteries de condensateurs** : la compensation de l'énergie réactive se dégrade, `Lagging_Current_Reactive.Power_kVarh` augmente et `Lagging_Current_Power_Factor` baisse.
  * **Pic de commandes** : le mix de production bascule vers la charge maximale (part de `Load_Type = Maximum_Load` doublée) et l'activité s'étend aux week-ends (`WeekStatus`), ce qui décale aussi les lags de consommation.
  * **Dérive saisonnière naturelle** : les données de novembre – décembre diffèrent de la période d'entraînement (janvier – août).
* **Méthode d'injection de la dérive :** Module `DriftSimulator` (`src/monitoring/`) qui génère des lots de production à partir de novembre – décembre en appliquant un décalage gaussien sur les grandeurs électriques et un rééchantillonnage des catégories, avec une intensité paramétrable (faible / modérée / critique).
* **Seuil d'alerte :** PSI $\ge 0.20$ sur une feature clé (puissance réactive, facteur de puissance, `lag_1`) ou proportion de features en dérive (KS / $\chi^2$, p-value < 0.05) $\ge 30\%$
* **Comportement autonome du système :** Le Flow Prefect de monitoring (planifié par CRON) calcule les statistiques, expose les métriques sur `/metrics` (alerte Prometheus) et génère le rapport HTML Evidently. En cas de dépassement du seuil, il déclenche le Flow de réentraînement incluant les données récentes ; le nouveau modèle (challenger) est comparé au champion sur la MAE, promu dans le Model Registry s'il est meilleur, puis déployé sur l'environnement inactif (green) avant bascule du trafic.

---

## 6. Planning Prévisionnel & Validation Enseignant

* [ ] **Jalon 1 (S2) :** Dépôt initialisé, Synopsis validé, Dataset importé sous DVC
* [ ] **Jalon 2 (S4) :** Pipeline DVC + MLflow fonctionnels, 1ers tests Pytest
* [ ] **Jalon 3 (S5) :** API FastAPI opérationnelle, Dockerfile validé, Workflow CI actif
* [ ] **Jalon 4 (S6) :** Flow Prefect orchestré, Détection de drift & Prometheus
* [ ] **Jalon Final (S8) :** Tag `v1.0-release`, Rapport PDF et Soutenance

> **Avis de l'Enseignant :** `[  ] VALIDÉ  /  [  ] À REVOIR`  
> **Commentaires :**
