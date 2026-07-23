# Patronising and Condescending Language Detection

Fine-tuning a pretrained DeBERTa-v3 NLI model for SemEval 2022 Task 4, Subtask 1: binary detection of patronising and condescending language (PCL).

The project treats each paragraph as the premise of a natural-language-inference pair and asks whether it entails the hypothesis:

> This text is patronising or condescending towards vulnerable people.

This formulation reuses an NLI-pretrained transformer for a subtle, highly imbalanced classification task.

## Training approach

The main experiment in `BestModel/bestmodel.ipynb` fine-tunes `cross-encoder/nli-deberta-v3-base` with:

- PyTorch and Hugging Face Transformers;
- focal loss with minority-class weighting;
- layer-wise learning-rate decay across DeBERTa encoder blocks;
- AdamW, linear warm-up, gradient clipping, and early stopping;
- deterministic seeds and cuDNN settings;
- explicit NaN/Inf guards in the ablation training path.

The stored run used 6,700 training paragraphs, 1,675 validation paragraphs, and a separate 2,093-example held-out development set. The positive class accounts for approximately 9.5% of each split.

## Recorded results

These values are taken from the saved notebook outputs and use a fixed decision threshold of 0.5:

| Split / metric | Value |
| --- | ---: |
| Best validation positive-class F1 | 0.5757 |
| Held-out development positive-class F1 | 0.5707 |
| Held-out development precision | 0.5637 |
| Held-out development recall | 0.5779 |
| Held-out development AUPRC | 0.6013 |
| Held-out development ROC-AUC | 0.9160 |
| Held-out development MCC | 0.5250 |

The held-out confusion matrix contains 115 true positives, 89 false positives, 1,805 true negatives, and 84 false negatives.

The notebook also records controlled three-epoch ablations on the same held-out split:

| Experiment | Held-out development F1 |
| --- | ---: |
| NLI formulation + weighted cross-entropy | 0.5596 |
| Plain DeBERTa-v3 binary classifier | 0.4082 |

These ablations isolate the value of the NLI task formulation. They use shorter training schedules than the five-epoch main run, so they should be read as within-notebook comparisons rather than universal benchmarks.

## Error analysis

`error_analysis.py` reloads the trained checkpoint, runs held-out inference, and exports every false positive and false negative with its confidence, keyword, country code, and source text. The notebook additionally analyses:

- precision-recall behaviour and class-prevalence baseline;
- confidence distributions for false positives and false negatives;
- per-keyword error patterns;
- MCC, Cohen's kappa, ROC-AUC, and false-positive/false-negative rates.

The stored run shows that many errors are high-confidence, which is documented as a model limitation rather than hidden behind a single aggregate score.

## Repository layout

| Path | Purpose |
| --- | --- |
| `BestModel/bestmodel.ipynb` | Main training, evaluation, ablations, and diagnostics |
| `Experiments/` | Earlier model and formulation experiments |
| `data_analysis.ipynb` | Dataset profiling, lexical analysis, and TF-IDF/SVD/t-SNE inspection |
| `error_analysis.py` | Re-runnable held-out error analysis |
| `dev.txt`, `test.txt` | Submitted prediction files |
| `requirements.txt` | Core numerical, plotting, and PyTorch dependencies |

The dataset and trained checkpoint are not committed, so the repository preserves the experiment code and notebook outputs rather than a self-contained model release.

## Reproduce the experiment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt jupyter transformers tqdm
jupyter lab
```

Place the official SemEval files at the paths used by the notebooks:

```text
Data/
├── Test/task4_test.tsv
└── TrainVal/
    ├── train_semeval_parids-labels.csv
    └── dev_semeval_parids-labels.csv
Official_DataSets/
└── dontpatronizeme_pcl.tsv
```

Then run `BestModel/bestmodel.ipynb` from top to bottom. The notebook writes its best checkpoint to `best_checkpoint_new/`.

## Scope

This project demonstrates end-to-end fine-tuning and evaluation of a pretrained transformer on a single GPU. It does not claim pretraining from scratch or distributed, foundation-model-scale training.
