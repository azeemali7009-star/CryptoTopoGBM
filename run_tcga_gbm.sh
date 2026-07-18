#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
python src/01_download_tcga_gbm_gdc.py --project TCGA-GBM --out data/raw/tcga_gbm
python src/02_preprocess.py --mode gdc --input data/raw/tcga_gbm --out data/processed_tcga_gbm --top-features 500
python src/03_build_complex.py --processed data/processed_tcga_gbm --out data/complex_tcga_gbm --k 12
python src/04_train_models.py --complex data/complex_tcga_gbm --processed data/processed_tcga_gbm --out results_tcga_gbm --seeds 7,19,31 --epochs 220
python src/05_paillier_encrypted_inference.py --model results_tcga_gbm/simplicial_model.pt --embeddings results_tcga_gbm/simplicial_embeddings.npy --out results_tcga_gbm/paillier_encrypted_inference.csv --n-examples 20 --bits 512
