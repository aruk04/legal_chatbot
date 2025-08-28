# eligibility_classifier.py
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
# Fixed "today" for reproducible demos (Asia/Kolkata date you used)
TODAY = datetime(2025, 8, 27)

# Resolve paths relative to this file, but allow passing explicit paths
BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "eligibility_data.jsonl"
MODEL_PATH = BASE / "models" / "eligibility_pipeline.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# IO + Preprocess
# ---------------------------------------------------------------------
def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Text & categorical
    df["text"] = df.get("text", "").fillna("").astype(str)
    df["intent"] = df.get("intent", "unknown").fillna("unknown").astype(str)
    df["use_context"] = df.get("use_context", "personal").fillna("personal").astype(str)

    # Numeric
    df["complaint_value"] = pd.to_numeric(df.get("complaint_value", 0), errors="coerce").fillna(0)

    # Dates
    df["purchase_date"] = pd.to_datetime(df.get("purchase_date", pd.NaT), errors="coerce")
    ds = (TODAY - df["purchase_date"]).dt.days
    df["days_since_purchase"] = ds.fillna(99999).astype(int)

    # Warranty
    df["warranty_period_months"] = pd.to_numeric(df.get("warranty_period_months", 0), errors="coerce").fillna(0).astype(int)
    df["within_warranty"] = (df["days_since_purchase"] <= (df["warranty_period_months"] * 30)).astype(int)

    # Limitation ~2 years
    df["within_limitation"] = (df["days_since_purchase"] <= 730).astype(int)

    # Scaled value
    df["value_log"] = np.log1p(df["complaint_value"])

    return df

def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Keep this in one place so train() and predict_*() are consistent
    cols = ["text","value_log","days_since_purchase","within_limitation","within_warranty","intent","use_context"]
    return df[cols]

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    text_col = "text"
    num_cols  = ["value_log","days_since_purchase","within_limitation","within_warranty"]
    cat_cols  = ["intent","use_context"]

    pre = ColumnTransformer(
        transformers=[
            ("txt", TfidfVectorizer(max_features=4000, ngram_range=(1,2)), text_col),
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    clf = LogisticRegression(max_iter=400, class_weight="balanced")
    return Pipeline([("feats", pre), ("clf", clf)])

# ---------------------------------------------------------------------
# Threshold tuning 
# ---------------------------------------------------------------------
def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Pick a decision threshold using Youden's J statistic on PR curve proxy:
    Choose threshold that maximizes (recall - precision gap penalty).
    You can swap this to ROC-based Youden J easily if needed.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Avoid NaNs/inf, pick argmax of (recall + precision - 1) ≈ PR Youden
    scores = recalls + precisions - 1.0
    # thresholds has len = len(precisions)-1; align
    if len(thresholds) == 0:
        return 0.5
    idx = int(np.nanargmax(scores[:-1])) if len(scores) > 1 else 0
    thr = float(np.clip(thresholds[idx], 0.05, 0.95))  # keep sane bounds
    return thr

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def _min_folds(y: pd.Series, desired: int = 5) -> int:
    """
    Ensure we don't ask for more folds than min class count.
    """
    min_class = int(y.value_counts().min())
    return max(2, min(desired, min_class))

def train(jsonl_path: Path = DATA_PATH, model_out: Path = MODEL_PATH, use_cv: bool = True, test_size: float = 0.2) -> Tuple[Pipeline, Optional[float]]:
    """
    Train the pipeline. By default:
      - Reports cross-validated metrics (safer for small datasets)
      - Fits final model on ALL data and saves it
      - Returns (pipeline, tuned_threshold or None)
    """
    jsonl_path = Path(jsonl_path)
    model_out = Path(model_out)

    df = read_jsonl(jsonl_path)
    df = preprocess(df)

    if "eligible_label" not in df.columns:
        raise ValueError("Dataset must include 'eligible_label' (0/1).")

    y = df["eligible_label"].astype(int).to_numpy()
    X = _feature_frame(df)
    pipe = build_pipeline()

    tuned_threshold = None

    if use_cv:
        k = _min_folds(pd.Series(y), desired=5)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        # Cross-validated predictions on the whole dataset
        y_pred = cross_val_predict(pipe, X, y, cv=skf, method="predict")
        try:
            y_proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:,1]
        except Exception:
            y_proba = None

        print("=== Cross-validated metrics (n_splits=%d) ===" % k)
        # zero_division=0 to avoid warnings on tiny/imbalanced folds
        print(classification_report(y, y_pred, zero_division=0))
        if y_proba is not None and len(np.unique(y)) == 2:
            try:
                print("ROC AUC (CV):", roc_auc_score(y, y_proba))
                tuned_threshold = tune_threshold(y, y_proba)
                print(f"Suggested decision threshold (from CV): {tuned_threshold:.2f}")
            except Exception:
                pass

        # Fit final model on ALL data
        pipe.fit(X, y)

    else:
        # Classic holdout (kept for completeness)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        pipe.fit(X_tr, y_tr)
        y_pr = pipe.predict(X_te)
        try:
            y_pb = pipe.predict_proba(X_te)[:,1]
        except Exception:
            y_pb = None

        print("=== Holdout metrics ===")
        print(classification_report(y_te, y_pr, zero_division=0))
        if y_pb is not None and len(np.unique(y_te)) == 2:
            try:
                print("ROC AUC:", roc_auc_score(y_te, y_pb))
                tuned_threshold = tune_threshold(y_te, y_pb)
                print(f"Suggested decision threshold (from holdout): {tuned_threshold:.2f}")
            except Exception:
                pass

    joblib.dump({"pipeline": pipe, "threshold": tuned_threshold}, model_out)
    print("Saved:", model_out)
    return pipe, tuned_threshold

# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------
def load_model(path: Path = MODEL_PATH):
    blob = joblib.load(path)
    # Backward compatibility: you might have old joblib with pipeline only
    if isinstance(blob, dict) and "pipeline" in blob:
        return blob
    return {"pipeline": blob, "threshold": None}

def _record_from_state(state: dict) -> dict:
    # tolerate missing values with safe defaults
    return {
        "text": state.get("text",""),
        "intent": state.get("intent","unknown"),
        "complaint_value": state.get("complaint_value") or 0,
        "purchase_date": state.get("purchase_date") or "1970-01-01",
        "warranty_period_months": state.get("warranty_period_months") or 0,
        "use_context": state.get("use_context") or "personal",
    }

def predict_proba(model_blob, state: dict) -> float:
    """
    model_blob is what load_model() returns: {"pipeline": Pipeline, "threshold": Optional[float]}
    """
    pipe: Pipeline = model_blob["pipeline"]
    rec = _record_from_state(state)
    df = preprocess(pd.DataFrame([rec]))
    X = _feature_frame(df)
    return float(pipe.predict_proba(X)[0,1])

def predict_label(model_blob, state: dict, threshold: Optional[float] = None) -> dict:
    """
    Use tuned threshold if available, else provided threshold, else 0.5.
    Returns: {"eligibility": "...", "confidence": proba, "threshold": used_threshold}
    """
    p = predict_proba(model_blob, state)
    used_thr = (
        model_blob.get("threshold")
        if model_blob.get("threshold") is not None
        else (threshold if threshold is not None else 0.5)
    )
    label = "Eligible" if p >= used_thr else "Not Eligible"
    return {"eligibility": label, "confidence": p, "threshold": used_thr}
