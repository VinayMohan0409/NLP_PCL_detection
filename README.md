# SemEval 2022 Task 4 - PCL Detection
**Vinay Udit Mohan | vum25@ic.ac.uk | CID: 02286845**

## Approach
PCL binary classification reframed as a Natural Language Inference (NLI) task using `cross-encoder/nli-deberta-v3-base`, fine-tuned with Focal Loss and Layer-wise Learning Rate Decay (LLRD). Full details in the report.

## Repository Structure

- **BestModel/**
  - `bestmodel.ipynb` — Full training pipeline (NLI formulation, Focal Loss, LLRD)
  - `best_checkpoint_new/` — Saved model weights

- **Experiments/** — Earlier experiments (multi-hypothesis, ablations)

- `dev.txt` — Predictions on official dev set (Exercise 5.1)
- `test.txt` — Predictions on official test set (Exercise 5.1)

- `data_analysis.ipynb` — EDA notebook (Stage 2)
- `error_analysis.py` — Error analysis script (Stage 5)
- `error_analysis.csv` — Error analysis output

- `requirements.txt`


## Reproducing Results
1. Install dependencies: `pip install -r requirements.txt`
2. Place data files in `Data/` and `Official_DataSets/` as expected by the notebook
3. Run `BestModel/bestmodel.ipynb` end-to-end

**Dev F1: 0.5707** | Threshold: 0.5 (fixed)

## Leaderboard Name
Vinay Udit Mohan
