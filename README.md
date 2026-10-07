# ML-SHM

Comparaison de modèles de détection d'endommagement (SHM, ondes guidées) sur plusieurs datasets.

## Principe

```
données brutes ──► split (1 fois / dataset, partagé par tous les modèles) ──► scaler ajusté sur train
      ──► tuning Optuna sur val ──► entraînements finaux (n seeds) ──► seuil sur val saine ──► test
```

| Ensemble | Contenu | Sert à |
|---|---|---|
| train | signaux sains uniquement | entraînement, ajustement du scaler |
| val | sains (autres blocs) + `damaged_val_size` des endommagés | early stopping, tuning, choix du seuil |
| test | sains + endommagés jamais vus | évaluation finale |

Un **bloc d'acquisition** (un fichier mensuel pour UTAH, `block_size` cycles consécutifs pour OGW)
n'est jamais partagé entre deux ensembles : des mesures voisines, presque identiques, des deux côtés
du split rendraient les scores trop optimistes.

## Utilisation

Depuis la racine du dépôt :

```bash
pip install -r requirements.txt

python src/main.py run   config/experiments/cae_utah.json            # tuning + entraînement + test
python src/main.py run   config/experiments/*.json                   # plusieurs expériences à la suite
python src/main.py tune  config/experiments/cae_utah.json            # tuning seul
python src/main.py train config/experiments/cae_utah.json            # réutilise best_params.json
python src/main.py plot  results/CAE_UTAH_pr_auc --seed 0            # figure de rapport
```

Le tableau de comparaison est `results/summary.csv` : une ligne par expérience
(PR-AUC, ROC-AUC, F1, précision, rappel, FPR sur le test, moyenne et écart-type sur les seeds).

## Configuration d'une expérience

Une config décrit **l'expérience** (modèle, dataset, split, budget), pas les hyperparamètres du modèle.
Toutes les clés sont optionnelles sauf `data.dataset` et `model.type` (valeurs par défaut dans `src/config.py`).

| Clé | Rôle |
|---|---|
| `data.params` | paramètres du loader (`nb_sample`, `path_index` pour UTAH ; `nb_cycles`, `freq_khz`, `channel`, `block_size`, `base_dir` pour OGW) |
| `split.strategy` | `group` (blocs tirés au hasard), `chrono` (anciens → train, récents → test), `random` (par signal, optimiste) |
| `model.params` | hyperparamètres **fixés** pour cette expérience (non réglés) |
| `model.search_space` | surcharge de l'espace de recherche du modèle |
| `tuning.objective` | `pr_auc` / `roc_auc` (sélection semi-supervisée, utilise les endommagés de val) ou `val_loss` (non supervisée) |
| `tuning.resume` | conserve les essais dans `results/<nom>/optuna.db` pour reprendre un tuning interrompu |
| `threshold` | `mean_std` (moyenne + k·écart-type, k = `value`) ou `quantile` (quantile `value` des scores sains de val) |
| `evaluation.n_seeds` | nombre d'entraînements finaux avec les meilleurs hyperparamètres |

## Ajouter un modèle

1. Créer une sous-classe de `BaseDetector` dans `src/models/` avec `DEFAULTS`, `SEARCH_SPACE`,
   `forward`, `training_step` (loss) et `anomaly_score` (un score par signal).
2. L'enregistrer dans `MODELS` (`src/models/__init__.py`).

La boucle d'entraînement, le tuning et l'évaluation sont génériques.

## Ajouter un dataset

1. Écrire un loader qui retourne un `SHMDataset` **brut** (`X`, `y`, `groups`, `meta`), sans normalisation.
2. L'enregistrer dans `DATASETS` (`src/data/__init__.py`).

## Sorties

```
results/
  summary.csv                       comparaison de toutes les expériences
  splits/<dataset>_<strategy>_<hash>.npz
  <expérience>/
    config.json  best_params.json  trials.csv  summary.json
    seed_<k>/  model.pt  scaler.pkl  history.json  metrics.json  scores.npz
```
