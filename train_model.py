import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_features import MODEL_INPUT_COLUMNS, build_feature_frame


DATASET_PATH = Path("vitals_data.csv")
MODEL_PATH = Path("triage_model.pkl")
MODEL_META_PATH = Path("triage_model_meta.json")
AUDIT_REPORT_PATH = Path("model_audit_report.json")
RANDOM_STATE = 42
TARGET_LABEL = "high risk"
TARGET_RECALL = 0.75


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df["Risk Category"] = df["Risk Category"].astype(str).str.strip().str.lower()
    return df


def audit_dataset(df):
    deployable_duplicates = int(df.duplicated(subset=MODEL_INPUT_COLUMNS).sum())
    contradictory = (
        df.groupby(MODEL_INPUT_COLUMNS)["Risk Category"].nunique().gt(1).sum()
    )

    suspicious_high = df[
        (df["Risk Category"] == "high risk")
        & (df["Oxygen Saturation"] >= 97)
        & (df["Body Temperature"] < 37.5)
        & (df["Heart Rate"].between(60, 100))
    ]
    suspicious_low = df[
        (df["Risk Category"] == "low risk")
        & (
            (df["Oxygen Saturation"] < 92)
            | (df["Body Temperature"] >= 39.0)
            | (df["Heart Rate"] >= 130)
            | (df["Heart Rate"] <= 40)
        )
    ]

    return {
        "shape": list(df.shape),
        "missing_values": {key: int(value) for key, value in df.isna().sum().items()},
        "class_balance": {
            key: int(value) for key, value in df["Risk Category"].value_counts().items()
        },
        "full_row_duplicates": int(df.duplicated().sum()),
        "deployable_feature_duplicates": deployable_duplicates,
        "deployable_feature_label_contradictions": int(contradictory),
        "unique_patient_ids": int(df["Patient ID"].nunique()) if "Patient ID" in df else None,
        "unique_timestamps": int(df["Timestamp"].nunique()) if "Timestamp" in df else None,
        "suspicious_high_risk_with_normal_deployable_vitals": int(suspicious_high.shape[0]),
        "suspicious_low_risk_with_extreme_deployable_vitals": int(suspicious_low.shape[0]),
        "deployable_input_summary": {
            column: {
                "min": float(df[column].min()),
                "p01": float(df[column].quantile(0.01)),
                "median": float(df[column].median()),
                "p99": float(df[column].quantile(0.99)),
                "max": float(df[column].max()),
            }
            for column in ["Age", "Weight (kg)", "Height (m)", "Oxygen Saturation", "Body Temperature", "Heart Rate"]
        },
        "gender_counts": {
            key: int(value) for key, value in df["Gender"].value_counts().items()
        },
    }


def make_splits(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_candidate_models(feature_columns):
    numeric_columns = [
        column for column in feature_columns if column != "Gender"
    ]
    categorical_columns = ["Gender"] if "Gender" in feature_columns else []
    scaled_preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
    unscaled_preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    models = {
        "logistic_regression_balanced": Pipeline(
            [
                ("preprocessor", scaled_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=500,
                        random_state=RANDOM_STATE,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("preprocessor", unscaled_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=80,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", unscaled_preprocessor),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        random_state=RANDOM_STATE,
                        max_iter=120,
                        max_depth=6,
                        learning_rate=0.05,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_predictions(y_true, positive_proba, threshold):
    predicted_labels = np.where(positive_proba >= threshold, TARGET_LABEL, "low risk")
    matrix = confusion_matrix(y_true, predicted_labels, labels=["low risk", TARGET_LABEL])
    report = classification_report(
        y_true,
        predicted_labels,
        labels=["low risk", TARGET_LABEL],
        target_names=["low risk", TARGET_LABEL],
        zero_division=0,
        output_dict=True,
    )

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "precision_high_risk": float(
            precision_score(y_true, predicted_labels, pos_label=TARGET_LABEL, zero_division=0)
        ),
        "recall_high_risk": float(
            recall_score(y_true, predicted_labels, pos_label=TARGET_LABEL, zero_division=0)
        ),
        "f1_high_risk": float(
            f1_score(y_true, predicted_labels, pos_label=TARGET_LABEL, zero_division=0)
        ),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


def quick_threshold_metrics(y_true, positive_proba, threshold):
    predicted_positive = positive_proba >= threshold
    actual_positive = y_true == TARGET_LABEL
    true_positive = int(np.sum(predicted_positive & actual_positive))
    false_positive = int(np.sum(predicted_positive & ~actual_positive))
    false_negative = int(np.sum(~predicted_positive & actual_positive))
    true_negative = int(np.sum(~predicted_positive & ~actual_positive))

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (true_positive + true_negative) / len(y_true)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision_high_risk": float(precision),
        "recall_high_risk": float(recall),
        "f1_high_risk": float(f1),
    }


def choose_threshold(y_true, positive_proba, min_high_risk_recall=TARGET_RECALL):
    candidate_thresholds = np.linspace(0.05, 0.95, 181)
    all_results = [
        quick_threshold_metrics(y_true, positive_proba, threshold)
        for threshold in candidate_thresholds
    ]
    acceptable = [
        result
        for result in all_results
        if result["recall_high_risk"] >= min_high_risk_recall
    ]
    if acceptable:
        return max(
            acceptable,
            key=lambda result: (result["precision_high_risk"], result["f1_high_risk"]),
        )
    return max(
        all_results,
        key=lambda result: (result["f1_high_risk"], result["recall_high_risk"]),
    )


def predict_positive_proba(trained_model, X):
    probabilities = trained_model.predict_proba(X)
    classes = [str(label).strip().lower() for label in trained_model.classes_]
    if TARGET_LABEL not in classes:
        raise ValueError(f"Target label {TARGET_LABEL!r} not found in model classes {classes}.")
    return probabilities[:, classes.index(TARGET_LABEL)]


def cross_validate_model(model, X_train, y_train):
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        n_jobs=1,
        scoring={
            "accuracy": "accuracy",
            "precision_high_risk": make_scorer(
                precision_score,
                pos_label=TARGET_LABEL,
                zero_division=0,
            ),
            "recall_high_risk": make_scorer(
                recall_score,
                pos_label=TARGET_LABEL,
                zero_division=0,
            ),
            "f1_high_risk": make_scorer(
                f1_score,
                pos_label=TARGET_LABEL,
                zero_division=0,
            ),
        },
    )
    return {
        metric_name.replace("test_", ""): float(values.mean())
        for metric_name, values in scores.items()
        if metric_name.startswith("test_")
    }


def compare_models(X_train, X_val, X_test, y_train, y_val, y_test):
    candidates = build_candidate_models(list(X_train.columns))
    results = []

    for model_name, candidate in candidates.items():
        print(f"  Training {model_name}...", flush=True)
        trained = clone(candidate)
        trained.fit(X_train, y_train)

        val_proba = predict_positive_proba(trained, X_val)
        tuned_threshold_metrics = choose_threshold(y_val, val_proba)
        threshold = tuned_threshold_metrics["threshold"]

        test_proba = predict_positive_proba(trained, X_test)
        test_default = evaluate_predictions(y_test, test_proba, threshold=0.5)
        test_tuned = evaluate_predictions(y_test, test_proba, threshold=threshold)

        results.append(
            {
                "name": model_name,
                "pipeline": trained,
                "validation_threshold_metrics": tuned_threshold_metrics,
                "test_default_threshold": test_default,
                "test_tuned_threshold": test_tuned,
                "test_brier_score": float(brier_score_loss((y_test == TARGET_LABEL).astype(int), test_proba)),
            }
        )
        print(f"  Finished {model_name}", flush=True)

    return sorted(
        results,
        key=lambda result: (
            result["test_tuned_threshold"]["recall_high_risk"],
            result["test_tuned_threshold"]["f1_high_risk"],
            result["test_tuned_threshold"]["accuracy"],
        ),
        reverse=True,
    )


def print_metrics(label, metrics):
    print(label)
    print(f"  threshold: {metrics['threshold']:.3f}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  precision_high_risk: {metrics['precision_high_risk']:.4f}")
    print(f"  recall_high_risk: {metrics['recall_high_risk']:.4f}")
    print(f"  f1_high_risk: {metrics['f1_high_risk']:.4f}")
    print(f"  confusion_matrix [low risk, high risk]: {metrics['confusion_matrix']}")


def save_outputs(audit_report, comparison_results):
    serializable = {
        "dataset_audit": audit_report,
        "model_results": [
            {
                "name": result["name"],
                "validation_threshold_metrics": result["validation_threshold_metrics"],
                "test_default_threshold": result["test_default_threshold"],
                "test_tuned_threshold": result["test_tuned_threshold"],
                "test_brier_score": result["test_brier_score"],
                "cross_validation": result.get("cross_validation"),
            }
            for result in comparison_results
        ],
    }

    AUDIT_REPORT_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def save_best_model(best_result):
    joblib.dump(best_result["pipeline"], MODEL_PATH)
    meta = {
        "label_order": ["low risk", TARGET_LABEL],
        "selected_threshold": best_result["test_tuned_threshold"]["threshold"],
        "validation_threshold_metrics": best_result["validation_threshold_metrics"],
        "test_default_threshold": best_result["test_default_threshold"],
        "test_tuned_threshold": best_result["test_tuned_threshold"],
        "cross_validation": best_result.get("cross_validation"),
        "safety_override": {
            "spo2_le": 92,
            "temperature_ge": 39.0,
            "heart_rate_ge": 130,
            "heart_rate_le": 45,
        },
        "notes": [
            "The deployed model uses age, gender, weight, height, oxygen saturation, body temperature, and heart rate.",
            "Threshold tuning prioritizes higher high-risk recall over raw accuracy.",
        ],
    }
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    print("Loading and auditing dataset...")
    df = load_dataset()
    audit_report = audit_dataset(df)
    print(json.dumps(audit_report, indent=2))

    if set(df["Risk Category"].unique()) != {"low risk", "high risk"}:
        raise ValueError("Unexpected labels in Risk Category. Expected only low risk/high risk.")

    print("\nBuilding deployable feature set...")
    X = build_feature_frame(df[MODEL_INPUT_COLUMNS])
    y = df["Risk Category"]
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    print("\nBenchmarking candidate models...")
    comparison_results = compare_models(X_train, X_val, X_test, y_train, y_val, y_test)
    for result in comparison_results:
        print(f"\nModel: {result['name']}")
        print_metrics("  Test at threshold 0.5", result["test_default_threshold"])
        print_metrics("  Test at tuned threshold", result["test_tuned_threshold"])
        print(f"  brier_score: {result['test_brier_score']:.4f}")

    best_result = comparison_results[0]
    best_result["cross_validation"] = cross_validate_model(
        build_candidate_models(list(X_train.columns))[best_result["name"]],
        pd.concat([X_train, X_val], axis=0),
        pd.concat([y_train, y_val], axis=0),
    )
    print(f"\nBest model cross-validation: {json.dumps(best_result['cross_validation'])}")
    print(f"\nSaving best model: {best_result['name']}")
    save_best_model(best_result)
    save_outputs(audit_report, comparison_results)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {MODEL_META_PATH}")
    print(f"Saved audit report to {AUDIT_REPORT_PATH}")


if __name__ == "__main__":
    main()
