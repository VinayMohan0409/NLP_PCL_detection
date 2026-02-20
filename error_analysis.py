import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

DATA_DIR      = Path("Data/TrainVal")
OFFICIAL_PATH = Path("Official_DataSets/dontpatronizeme_pcl.tsv")
CHECKPOINT_DIR = Path("best_checkpoint_new")

MODEL_NAME  = "cross-encoder/nli-deberta-v3-base"
MAX_LENGTH  = 128
EVAL_BATCH  = 32
THRESHOLD   = 0.5
HYPOTHESES  = [
    "This text is patronising or condescending towards vulnerable people.",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Data loading 
def load_data():
    official_cols = ["par_id", "article_id", "keyword",
                     "country_code", "text", "orig_dataset_label"]
    official_df = pd.read_csv(
        OFFICIAL_PATH, sep="\t", names=official_cols, skiprows=4
    )
    official_df["par_id"] = official_df["par_id"].astype(int)

    train_labels_df = pd.read_csv(DATA_DIR / "train_semeval_parids-labels.csv")
    dev_labels_df   = pd.read_csv(DATA_DIR / "dev_semeval_parids-labels.csv")

    def to_binary(label_str):
        return int(any(ast.literal_eval(label_str)))

    for df in (train_labels_df, dev_labels_df):
        df["label_binary"] = df["label"].apply(to_binary)

    def merge(labels_df):
        return labels_df.merge(
            official_df[["par_id", "text", "keyword", "country_code"]],
            on="par_id", how="left"
        )

    dev_df = merge(dev_labels_df)
    return dev_df


# Inference 
@torch.no_grad()
def predict_with_ensemble(model, tokenizer, texts, hypotheses,
                           entailment_idx, max_length=MAX_LENGTH,
                           batch_size=EVAL_BATCH):
    model.eval()
    all_per_hyp = []

    for hyp in hypotheses:
        scores = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) if t is not None else "" for t in texts[i: i + batch_size]]
            enc = tokenizer(
                batch_texts,
                [hyp] * len(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(DEVICE)
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1)
            scores.append(probs[:, entailment_idx].cpu().numpy())
        all_per_hyp.append(np.concatenate(scores))

    per_hyp = np.stack(all_per_hyp, axis=0)   # (n_hyp, n_texts)
    ensemble = per_hyp.max(axis=0)   # max across hypotheses
    return ensemble, per_hyp


# Main 
def main():
    print("Loading data...")
    dev_df = load_data()
    dev_texts  = dev_df["text"].tolist()
    dev_labels = dev_df["label_binary"].values

    print(f"Dev set: {len(dev_df)} examples  "
          f"(PCL={dev_labels.sum()}, Not PCL={(dev_labels==0).sum()})")

    print(f"\nLoading checkpoint from {CHECKPOINT_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    model.to(DEVICE)

    # Resolve entailment index
    entailment_idx = 1
    for idx, name in model.config.id2label.items():
        if "entail" in name.lower():
            entailment_idx = int(idx)
    print(f"Entailment index: {entailment_idx}  "
          f"(id2label: {model.config.id2label})")

    print("\nRunning inference on dev set...")
    dev_scores, _ = predict_with_ensemble(
        model, tokenizer, dev_texts, HYPOTHESES, entailment_idx
    )

    dev_preds = (dev_scores >= THRESHOLD).astype(int)

    # Build results dataframe 
    results_df = dev_df[["par_id", "text", "keyword", "country_code"]].copy()
    results_df["true_label"]  = dev_labels
    results_df["pred_label"]  = dev_preds
    results_df["confidence"]  = dev_scores   # P(entailment)
    results_df["correct"]     = dev_labels == dev_preds

    errors_df = results_df[~results_df["correct"]].copy()

    # Error type
    def error_type(row):
        if row["true_label"] == 1 and row["pred_label"] == 0:
            return "False Negative (missed PCL)"
        return "False Positive (wrongly flagged)"

    errors_df["error_type"] = errors_df.apply(error_type, axis=1)

    # Sort: highest confidence first 
    errors_df = errors_df.sort_values("confidence", ascending=False)

    # Save
    out_path = Path("error_analysis.csv")
    errors_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(errors_df)} errors = {out_path}")

    # Summary printout
    fn = errors_df[errors_df["error_type"].str.startswith("False Neg")]
    fp = errors_df[errors_df["error_type"].str.startswith("False Pos")]

    overall_f1 = compute_f1(dev_labels, dev_preds)
    print(f"\n{'='*60}")
    print(f" Dev F1 (positive class): {overall_f1:.4f}")
    print(f" Total errors : {len(errors_df)} / {len(dev_df)}")
    print(f" False Negatives (missed PCL) : {len(fn)}")
    print(f" False Positives (wrongly flagged) : {len(fp)}")
    print(f"{'='*60}")

    # High-confidence errors (model was very wrong)
    high_conf_errors = errors_df[
        ((errors_df["error_type"].str.startswith("False Neg")) & (errors_df["confidence"] < 0.2)) |
        ((errors_df["error_type"].str.startswith("False Pos")) & (errors_df["confidence"] > 0.8))
    ]
    print(f"\n High-confidence errors (hardest failures): {len(high_conf_errors)}")

    # Keyword breakdown of errors
    print(f"\nKeyword breakdown of FALSE NEGATIVES")
    if len(fn) > 0:
        kw_fn = fn["keyword"].value_counts().head(10)
        print(kw_fn.to_string())

    print(f"\nKeyword breakdown of FALSE POSITIVES")
    if len(fp) > 0:
        kw_fp = fp["keyword"].value_counts().head(10)
        print(kw_fp.to_string())

    # Print 10 most embarrassing FNs and FPs for quick reading
    print(f"\n{'='*60}")
    print("TOP 10 HIGH-CONFIDENCE FALSE NEGATIVES")
    print(f"{'='*60}")
    fn_sorted = fn.sort_values("confidence", ascending=True).head(10)
    for _, row in fn_sorted.iterrows():
        print(f"\n  [keyword: {row['keyword']}]  conf={row['confidence']:.3f}")
        print(f"  {row['text'][:300]}")

    print(f"\n{'='*60}")
    print("TOP 10 HIGH-CONFIDENCE FALSE POSITIVES")
    print(f"{'='*60}")
    fp_sorted = fp.sort_values("confidence", ascending=False).head(10)
    for _, row in fp_sorted.iterrows():
        print(f"\n  [keyword: {row['keyword']}]  conf={row['confidence']:.3f}")
        print(f"  {row['text'][:300]}")

    print(f"\n Completed! Open error_analysis.csv for the full error table.")


def compute_f1(true, pred):
    tp = ((pred == 1) & (true == 1)).sum()
    fp = ((pred == 1) & (true == 0)).sum()
    fn = ((pred == 0) & (true == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


if __name__ == "__main__":
    main()