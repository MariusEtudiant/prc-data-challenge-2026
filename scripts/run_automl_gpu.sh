#!/usr/bin/env bash
# Free the AutoGluon model directories (predictions and leaderboards are kept),
# install torch in the isolated venv, then rerun the search WITH its neural models
# on the two airports that carry most of the error.
set -e
cd "$(dirname "$0")/.."
echo "=== nettoyage des artefacts (predictions conservees) ==="
du -sh experiments/automl
rm -rf experiments/automl/ag_*
df -h . | tail -1
echo "=== installation de torch dans le venv isole ==="
.venv_automl/Scripts/python.exe -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -2
.venv_automl/Scripts/python.exe -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "=== recherche AVEC reseaux, sur GPU ==="
for a in LIRF LFPG; do
  echo "----- $a -----"
  .venv_automl/Scripts/python.exe -u scripts/exp_m45_automl.py "$a" 25 --gpu 2>&1 \
    | grep -E "entrainement=|^ *(WeightedEnsemble|ExtraTrees|RandomForest|CatBoost|LightGBM|NeuralNet)|holdout NETTOYE"
done
