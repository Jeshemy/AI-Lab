from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cybersecurity_intrusion_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "safe_flow_model.joblib"
EVALUATION_PATH = MODEL_DIR / "evaluation.json"
SOURCE_URL = "https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset"
TARGET = "attack_detected"
ID_COLUMN = "session_id"
NUMERIC_FEATURES = [
    "network_packet_size", "login_attempts", "session_duration",
    "ip_reputation_score", "failed_logins", "unusual_time_access",
]
CATEGORICAL_FEATURES = ["protocol_type", "encryption_used", "browser_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FEATURE_TYPES = [
    {"name": feature, "type": "Numeric"} for feature in NUMERIC_FEATURES
] + [
    {"name": feature, "type": "Categorical"} for feature in CATEGORICAL_FEATURES
]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def metric_summary(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def aggregate_feature_importance(calibrated_model):
    """Average encoded-feature importances across the calibrated RF folds."""
    totals = {feature: 0.0 for feature in FEATURES}
    estimators = calibrated_model.calibrated_classifiers_
    for calibrated in estimators:
        pipeline = calibrated.estimator
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        values = pipeline.named_steps["classifier"].feature_importances_
        for encoded_name, value in zip(names, values):
            raw_name = encoded_name.split("__", 1)[-1]
            feature = next((item for item in FEATURES if raw_name == item or raw_name.startswith(item + "_")), None)
            if feature:
                totals[feature] += float(value) / len(estimators)
    denominator = sum(totals.values()) or 1.0
    return [
        {"name": feature, "importance": round(value / denominator, 6)}
        for feature, value in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def make_pipeline(classifier):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    frame = pd.read_csv(DATA_PATH, keep_default_na=False)
    for column in FEATURES + [TARGET]:
        frame[column] = frame[column].replace(r"^\s*$", pd.NA, regex=True)
    required = set(FEATURES + [TARGET, ID_COLUMN])
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    if frame.empty:
        raise ValueError("Dataset contains no records.")

    X = frame[FEATURES].copy()
    y = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("attack_detected must contain both binary labels 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", min_samples_leaf=3, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            C=1.0, kernel="rbf", class_weight="balanced", probability=True,
            random_state=RANDOM_STATE,
        ),
    }
    scoring = {
        "attack_recall": "recall",
        "attack_f1": "f1",
        "attack_f2": make_scorer(fbeta_score, beta=2, pos_label=1, zero_division=0),
        "accuracy": "accuracy",
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    fitted_candidates = {}
    comparison = []

    for name, classifier in candidates.items():
        pipeline = make_pipeline(classifier)
        cv_scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1,
            error_score="raise",
        )
        pipeline.fit(X_train, y_train)
        test_metrics = metric_summary(y_test, pipeline.predict(X_test))
        record = {
            "name": name,
            "cv_attack_recall": round(float(cv_scores["test_attack_recall"].mean()), 6),
            "cv_attack_f1": round(float(cv_scores["test_attack_f1"].mean()), 6),
            "cv_attack_f2": round(float(cv_scores["test_attack_f2"].mean()), 6),
            "cv_accuracy": round(float(cv_scores["test_accuracy"].mean()), 6),
            **test_metrics,
        }
        comparison.append(record)
        fitted_candidates[name] = pipeline
        print(
            f"{name}: CV attack recall={record['cv_attack_recall']:.4f}, "
            f"CV attack F2={record['cv_attack_f2']:.4f}, "
            f"CV F1={record['cv_attack_f1']:.4f}, test recall={record['recall']:.4f}, "
            f"test F1={record['f1']:.4f}"
        )

    comparison.sort(
        key=lambda item: (item["cv_attack_f2"], item["cv_attack_recall"], item["cv_accuracy"]),
        reverse=True,
    )
    best_f2 = comparison[0]["cv_attack_f2"]
    near_best = [item for item in comparison if item["cv_attack_f2"] >= best_f2 - 0.01]
    selected_name = max(
        near_best,
        key=lambda item: (item["cv_attack_f1"], item["cv_accuracy"], item["cv_attack_recall"]),
    )["name"]

    selected = CalibratedClassifierCV(
        estimator=make_pipeline(candidates[selected_name]),
        method="sigmoid",
        cv=cv,
        ensemble=False,
    )
    selected.fit(X_train, y_train)
    selected_metrics = metric_summary(y_test, selected.predict(X_test))
    selected_record = next(item for item in comparison if item["name"] == selected_name)
    feature_importance = aggregate_feature_importance(selected)


    baseline = {}
    for feature in NUMERIC_FEATURES:
        baseline[feature] = float(pd.to_numeric(X_train[feature], errors="coerce").median())
    for feature in CATEGORICAL_FEATURES:
        modes = X_train[feature].dropna().mode()
        baseline[feature] = str(modes.iloc[0]) if not modes.empty else "Unknown"

    class_counts = {"normal": int((y == 0).sum()), "attack": int((y == 1).sum())}
    all_missing_cells = int(frame[FEATURES + [TARGET]].isna().sum().sum())
    evaluation = {
        "dataset": {
            "name": "Cybersecurity Intrusion Detection Dataset",
            "source_url": SOURCE_URL,
            "license": "MIT",
            "records": int(len(frame)),
            "predictor_features": int(len(FEATURES)),
            "features": FEATURES,
            "feature_types": FEATURE_TYPES,
            "target": TARGET,
            "class_counts": class_counts,
            "missing_values": all_missing_cells,
            "duplicate_rows": int(frame.duplicated().sum()),
            "dataset_tagged_synthetic": True,
        },
        "preprocessing": [
            "Excluded session_id from model predictors to avoid identifier leakage.",
            "Imputed numeric values with the training median and standardized them.",
            "Imputed categorical values with the training mode and one-hot encoded them.",
            "Used stratified sampling to preserve normal/attack proportions.",
        ],
        "split": {
            "method": "Stratified 80/20 train/test holdout",
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "random_state": RANDOM_STATE,
            "selection_cv": "3-fold stratified cross-validation on training data",
            "selection_rule": "Keep candidates within 0.01 mean attack F2 (beta=2, recall-weighted) of the training-CV leader; select the highest mean attack F1, then accuracy, then recall.",
        },
        "selected_model": selected_name,
        "probability_calibration": "Sigmoid calibration fitted with 3-fold stratified cross-validation on training data only.",
        "model_selection_metric": "Attack F2 (beta=2; recall-weighted)",
        "feature_importance": feature_importance,
        "comparison_note": "Classifier comparison test metrics use each uncalibrated candidate's default decision rule for a like-for-like comparison. Selected-model results above use the calibrated final model.",
        "selected_cv": {
            "attack_recall": selected_record["cv_attack_recall"],
            "attack_f1": selected_record["cv_attack_f1"],
            "attack_f2": selected_record["cv_attack_f2"],
            "accuracy": selected_record["cv_accuracy"],
        },
        "test_metrics": {
            key: selected_metrics[key]
            for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix")
        },
        "comparison": comparison,
        "positive_class": "Attack (attack_detected = 1)",
        "confusion_matrix_labels": ["Normal", "Attack"],
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": selected,
        "features": FEATURES,
        "baseline": baseline,
        "feature_importance": feature_importance,
    }, MODEL_PATH)
    EVALUATION_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Selected model: {selected_name}")
    print(f"Test metrics: {json.dumps(evaluation['test_metrics'])}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved evaluation: {EVALUATION_PATH}")


if __name__ == "__main__":
    main()
