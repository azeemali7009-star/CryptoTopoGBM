# CryptoTopoGBM Revised Package

**Topic:** Privacy-preserving topological deep learning for glioblastoma survival-risk prediction from multi-omics profiles.

This revised package fixes the main weakness of the previous draft: it now contains numerical outputs, result tables, figures, ablation baselines, and an actual homomorphic-encryption final-head inference test.

## What is included

- `manuscript/main.tex` - revised LaTeX manuscript.
- `manuscript/main.pdf` - compiled PDF.
- `manuscript/figures/` - manuscript figures in PDF and PNG.
- `manuscript/tables/` - manuscript-ready LaTeX tables.
- `src/` - complete implementation code.
- `data/demo/gbm_like_multiomics_demo.csv` - executable GBM-like benchmark for local testing only.
- `data/processed/` - processed feature, label, and survival files from the benchmark.
- `data/complex/` - patient graph, triangles, normalized graph adjacency, and triangle-coface adjacency.
- `results/` - metrics, encrypted inference outputs, trained model checkpoint, and embeddings.

## Important scientific note

The bundled benchmark is synthetic and is included so the full method can be run without redistributing controlled or bulky TCGA patient files. It validates the computational workflow and the cryptographic inference mechanism. It is **not** a biological claim about glioblastoma.

For publishable TCGA results, run the GDC downloader and then run the same downstream scripts in GDC mode.

## Quick demo run

```bash
cd CryptoTopoGBM_Revised
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

python src/00_generate_gbm_like_demo.py --out data/demo/gbm_like_multiomics_demo.csv --n 240 --seed 7
python src/02_preprocess.py --mode demo --input data/demo/gbm_like_multiomics_demo.csv --out data/processed --top-features 300
python src/03_build_complex.py --processed data/processed --out data/complex --k 10
python src/04_train_models.py --complex data/complex --processed data/processed --out results --seeds 7,19,31 --epochs 80 --hidden-dim 48 --embedding-dim 24
python src/05_paillier_encrypted_inference.py --model results/simplicial_model.pt --embeddings results/simplicial_embeddings.npy --out results/paillier_encrypted_inference.csv --n-examples 8 --bits 256
python src/06_make_figures.py
```

## TCGA-GBM open-access run

```bash
export PYTHONPATH=src
python src/01_download_tcga_gbm_gdc.py --project TCGA-GBM --out data/raw/tcga_gbm
python src/02_preprocess.py --mode gdc --input data/raw/tcga_gbm --out data/processed_tcga_gbm --top-features 500
python src/03_build_complex.py --processed data/processed_tcga_gbm --out data/complex_tcga_gbm --k 12
python src/04_train_models.py --complex data/complex_tcga_gbm --processed data/processed_tcga_gbm --out results_tcga_gbm --seeds 7,19,31 --epochs 220
python src/05_paillier_encrypted_inference.py --model results_tcga_gbm/simplicial_model.pt --embeddings results_tcga_gbm/simplicial_embeddings.npy --out results_tcga_gbm/paillier_encrypted_inference.csv --n-examples 20 --bits 512
```

Optional CKKS/TenSEAL deployment can be added by installing TenSEAL and replacing the Paillier final-head script with the CKKS vector dot-product interface. The manuscript describes this as the CKKS-ready path; the package ships a dependency-free Paillier implementation to guarantee an encrypted-output table on any standard Python installation.

## Current executable benchmark result

The current bundled benchmark produced:

- SimplicialNet test AUC: `0.699 ± 0.056`
- GraphNet test AUC: `0.670 ± 0.095`
- MLP test AUC: `0.672 ± 0.044`
- Paillier encrypted final-head mean absolute logit error: `0.000114`

See:

- `results/metrics_summary_test.csv`
- `results/paillier_encrypted_summary.csv`
- `manuscript/main.pdf`

