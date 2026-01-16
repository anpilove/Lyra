# LYRA: Lyrics Retrieval and Annotation

Automatic annotation of Russian rap lyrics using retrieval baselines and generative models.

## Overview

LYRA builds a dataset of annotated lyric fragments from Genius and compares multiple approaches for producing explanations:

- Retrieval baselines: TF-IDF, BM25, SBERT, Hybrid, Ensemble
- Generative models: ruT5, ruT5 with RAG (BM25 + generation)

Dataset size: 3,291 songs with 22,220 annotations.

## Requirements

- Python 3.10+
- `requirements.txt`

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## Data

Main files:

- `data/annotations_dataset_new.json` (JSON)
- `data/annotations_dataset_new.jsonl` (JSONL, incremental output from collector)

## Collect annotations (optional)

The collector reads song metadata from `data/russian_song_lyrics.csv` and fetches annotations by `song_id`:

```bash
export GENIUS_TOKEN='YOUR_ACCESS_TOKEN'
python src/collect_annotations.py
```

Outputs:

- `data/annotations_dataset_new.json`
- `data/annotations_dataset_new.jsonl`

## Notebooks

All model code is self-contained in notebooks:

1. `notebooks/00_dataset_eda.ipynb` - dataset EDA
2. `notebooks/01_tfidf_baseline.ipynb`
3. `notebooks/02_bm25_retrieval.ipynb`
4. `notebooks/03_sbert_retrieval.ipynb` (GPU recommended)
5. `notebooks/04_hybrid_retrieval.ipynb` (GPU recommended)
6. `notebooks/05_results_comparison.ipynb`
7. `notebooks/06_rut5_generation.ipynb` (GPU recommended)
8. `notebooks/07_rag_rut5.ipynb` (GPU recommended)

Each notebook uses `MAX_EXAMPLES` or `MAX_SAMPLES` to limit runtime. Set to `None` for a full run.

## Evaluation scripts

Optional CLI runs are available:

```bash
python evaluate_all_approaches.py
python compare_approaches.py
python optimize_bm25.py
```

Saved metrics are written to `data/*.json`.

## Results

Retrieval metrics (ROUGE on a 2,000-example subset):

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | --- | --- | --- |
| TF-IDF | 0.0070 | 0.0027 | 0.0062 |
| BM25 | 0.0140 | 0.0019 | 0.0124 |
| SBERT | 0.0087 | 0.0050 | 0.0087 |
| Hybrid (alpha=0.5) | 0.0102 | 0.0050 | 0.0099 |
| Ensemble (Equal) | 0.0104 | 0.0043 | 0.0102 |

Generative metrics (same subset):

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | --- | --- | --- |
| ruT5 | 0.232 | 0.206 | 0.224 |
| ruT5 + RAG (BM25) | 0.279 | 0.251 | 0.271 |

Metrics files:

- `data/evaluation_results.json`
- `data/bm25_results.json`
- `data/sbert_results.json`
- `data/hybrid_results_alpha3.json`
- `data/hybrid_results_alpha5.json`
- `data/hybrid_results_alpha7.json`
- `data/ensemble_results.json`
- `data/sbert_models_comparison.json`

## Report

- `report.tex`
- `report.pdf`

## Project structure

```
Lyra/
├── src/                       # Data collection and legacy baselines
├── notebooks/                 # Self-contained model notebooks
├── data/                      # Datasets and evaluation results
├── report.tex / report.pdf    # Final report
```

## License

MIT. See `LICENSE`.
