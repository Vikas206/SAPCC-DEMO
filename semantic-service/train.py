"""
Phase 2 training: builds the calibrated conflict-probability model from real
outcome logs produced by Phase 1's Go engine running the real YCSB (A,B,C,D,F)
and TPC-C workloads (data/train.*.jsonl, collected with threshold=1.1 so
every transaction was actually attempted under OCC and has a ground-truth
conflict label -- no selection bias from a routing decision).

Pipeline:
  1. Load all labeled outcome rows.
  2. Feature extraction -- this stands in for the LLM semantic layer. It is
     intentionally swappable: `extract_features` takes the transaction's
     natural-language SemanticText (and type/key-count) and returns a
     feature vector. Right now that's TF-IDF + numeric features (an
     "LLM-ready, locally substituted" classifier, honest to call it that in
     the review -- see README). Swapping in a real LLM call means replacing
     only this function; everything downstream (logistic model, Platt
     scaling, calibration gate) is unchanged.
  3. Logistic regression on those features -> raw conflict probability.
  4. Platt scaling (sklearn's CalibratedClassifierCV, method="sigmoid" IS
     Platt scaling) on top, so the output probability is trustworthy, not
     just rank-ordered.
  5. Reliability diagram + Brier score on a held-out test split, saved as
     an image -- this is the evidence that "calibrated" isn't a buzzword.
"""
import glob
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from features import select_numeric


def load_training_rows(pattern="../data/train.*.jsonl"):
    rows = []
    for path in glob.glob(pattern):
        workload = path.split("train.")[1].replace(".jsonl", "")
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if not r["labeled"]:
                    continue  # only OCC-attempted txns have a ground-truth label
                r["workload"] = workload
                rows.append(r)
    return pd.DataFrame(rows)


def build_pipeline():
    # Numeric features (num_keys, num_writes) + TF-IDF over the semantic
    # text description -- the stand-in for what an LLM would extract from
    # reading the transaction's intent (see module docstring).
    text_feat = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=200, ngram_range=(1, 2))),
    ])
    pre = ColumnTransformer([
        ("text", text_feat, "text"),
        ("num", FunctionTransformer(select_numeric), ["num_keys", "num_writes"]),
    ])
    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    # Platt scaling: CalibratedClassifierCV(method="sigmoid") fits exactly
    # the logistic-on-the-score calibration Platt (1999) describes.
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    return Pipeline([("pre", pre), ("clf", calibrated)])


def main():
    df = load_training_rows()
    print(f"Loaded {len(df)} labeled rows across workloads: {df['workload'].value_counts().to_dict()}")
    print(f"Overall conflict rate: {df['conflict'].mean():.3f}")

    X = df[["text", "num_keys", "num_writes"]]
    y = df["conflict"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    p_test = pipe.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, p_test)
    auc = roc_auc_score(y_test, p_test) if y_test.nunique() > 1 else float("nan")
    print(f"Held-out Brier score: {brier:.4f}  (0=perfect, 0.25=uninformative)")
    print(f"Held-out ROC-AUC: {auc:.4f}")

    # Reliability diagram: predicted probability vs. observed conflict rate.
    frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#c9622f", label="SAPCC calibration model")
    ax.set_xlabel("Predicted conflict probability")
    ax.set_ylabel("Observed conflict rate")
    ax.set_title(f"Reliability diagram (Brier={brier:.4f}, AUC={auc:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig("../data/reliability_diagram.png", dpi=150)
    print("Saved ../data/reliability_diagram.png")

    with open("model.pkl", "wb") as f:
        pickle.dump(pipe, f)
    print("Saved model.pkl")

    # Per-type sanity check: does the model recover something close to the
    # hand-coded TPCCSemanticPredictor's intuition (NewOrder/Payment/Delivery
    # high, OrderStatus/StockLevel ~0) PURELY from data, with no type field
    # passed in -- only the natural-language text?
    print("\nPer-semantic-type mean predicted probability (learned from text alone):")
    df_test = df.loc[X_test.index].copy()
    df_test["pred"] = p_test
    print(df_test.groupby("type")["pred"].mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
