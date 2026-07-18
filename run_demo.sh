#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
python src/00_generate_gbm_like_demo.py --out data/demo/gbm_like_multiomics_demo.csv --n 240 --seed 7
python src/02_preprocess.py --mode demo --input data/demo/gbm_like_multiomics_demo.csv --out data/processed --top-features 300
python src/03_build_complex.py --processed data/processed --out data/complex --k 10
python src/04_train_models.py --complex data/complex --processed data/processed --out results --seeds 7,19,31 --epochs 80 --hidden-dim 48 --embedding-dim 24
python src/05_paillier_encrypted_inference.py --model results/simplicial_model.pt --embeddings results/simplicial_embeddings.npy --out results/paillier_encrypted_inference.csv --n-examples 8 --bits 256
python src/06_make_figures.py
