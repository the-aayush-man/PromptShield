from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


LABELS = [
    "Safe Prompt",
    "Prompt Injection",
    "Jailbreak",
    "Data Extraction",
    "Roleplay Manipulation",
]

RANDOM_STATE = 42


def load_split(path: Path) -> pd.DataFrame:
    required_columns = {"prompt", "label", "source_dataset"}
    frame = pd.read_csv(
        path,
        usecols=list(required_columns),
        dtype="string",
    )
    if set(frame.columns) != required_columns:
        raise ValueError(
            f"{path} must contain {sorted(required_columns)}; "
            f"found {list(frame.columns)}"
        )
    if frame.isna().any().any():
        raise ValueError(f"{path} contains missing required values")

    invalid_labels = sorted(set(frame["label"]) - set(LABELS))
    if invalid_labels:
        raise ValueError(f"{path} contains invalid labels: {invalid_labels}")
    return frame


def sqrt_balanced_weights(labels: pd.Series) -> dict[str, float]:
    counts = labels.value_counts()
    sample_count = len(labels)
    class_count = len(counts)
    return {
        str(label): math.sqrt(sample_count / (class_count * count))
        for label, count in counts.items()
    }


def make_classifier(
    class_weight: str | dict[str, float],
    *,
    c_value: float,
    max_iterations: int,
    tolerance: float,
    solver: str,
) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        max_iter=max_iterations,
        random_state=RANDOM_STATE,
        solver=solver,
        l1_ratio=0,
        tol=tolerance,
    )


def probability_metrics(
    model: LogisticRegression,
    features: Any,
) -> tuple[list[str], list[float], list[float]]:
    probabilities = model.predict_proba(features)
    class_names = list(model.classes_)
    predicted_indexes = probabilities.argmax(axis=1)
    predictions = [class_names[index] for index in predicted_indexes]
    confidence = probabilities.max(axis=1).tolist()

    sorted_probabilities = np.sort(probabilities, axis=1)
    margins = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    ).tolist()
    return predictions, confidence, margins


def evaluate_predictions(
    true_labels: pd.Series,
    predictions: list[str],
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    evaluation_labels = labels or LABELS
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predictions,
        labels=evaluation_labels,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(evaluation_labels)
    }
    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_f1": float(
            f1_score(
                true_labels,
                predictions,
                labels=evaluation_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_labels,
                predictions,
                labels=evaluation_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": per_class,
    }


def evaluate_by_source(
    frame: pd.DataFrame,
    predictions: list[str],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    prediction_series = pd.Series(predictions, index=frame.index)
    for source, group in frame.groupby("source_dataset"):
        source_predictions = prediction_series.loc[group.index].tolist()
        source_labels = [
            label for label in LABELS if label in set(group["label"])
        ]
        results[str(source)] = {
            "rows": len(group),
            **evaluate_predictions(
                group["label"],
                source_predictions,
                labels=source_labels,
            ),
        }
    return results


def save_evaluation_files(
    *,
    output_dir: Path,
    split_name: str,
    frame: pd.DataFrame,
    predictions: list[str],
    confidence: list[float],
    margins: list[float],
    metrics: dict[str, Any],
) -> None:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    prediction_frame = frame.copy()
    prediction_frame["prediction"] = predictions
    prediction_frame["confidence"] = confidence
    prediction_frame["confidence_margin"] = margins
    prediction_frame["correct"] = (
        prediction_frame["label"] == prediction_frame["prediction"]
    )
    prediction_frame.to_csv(split_dir / "predictions.csv", index=False)

    errors = prediction_frame[~prediction_frame["correct"]].sort_values(
        ["confidence", "confidence_margin"],
        ascending=False,
    )
    errors.head(200).to_csv(
        split_dir / "highest_confidence_errors.csv",
        index=False,
    )

    matrix = confusion_matrix(
        frame["label"],
        predictions,
        labels=LABELS,
    )
    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"actual::{label}" for label in LABELS],
        columns=[f"predicted::{label}" for label in LABELS],
    )
    matrix_frame.to_csv(split_dir / "confusion_matrix.csv")

    report_frame = pd.DataFrame(
        classification_report(
            frame["label"],
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_frame.to_csv(split_dir / "classification_report.csv")

    (split_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def save_top_features(
    *,
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    output_path: Path,
    feature_count: int = 30,
) -> None:
    feature_names = vectorizer.get_feature_names_out()
    rows: list[dict[str, Any]] = []

    for class_index, label in enumerate(model.classes_):
        coefficients = model.coef_[class_index]
        top_indexes = coefficients.argsort()[-feature_count:][::-1]
        for rank, feature_index in enumerate(top_indexes, start=1):
            rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "feature": feature_names[feature_index],
                    "coefficient": float(coefficients[feature_index]),
                }
            )

    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate PromptShield's TF-IDF Logistic Regression "
            "classifier."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("prepared_datasets"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_artifacts"),
    )
    parser.add_argument("--max-features", type=int, default=60000)
    parser.add_argument("--max-iterations", type=int, default=450)
    parser.add_argument("--c-value", type=float, default=2.0)
    parser.add_argument("--tolerance", type=float, default=0.002)
    parser.add_argument(
        "--solver",
        choices=["lbfgs", "saga"],
        default="lbfgs",
    )
    parser.add_argument(
        "--weighting",
        choices=["auto", "balanced", "sqrt_balanced"],
        default="auto",
    )
    args = parser.parse_args()

    started_at = time.time()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(data_dir / "train.csv")
    validation = load_split(data_dir / "validation.csv")
    test = load_split(data_dir / "test.csv")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=args.max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    print("Fitting TF-IDF vectorizer...", flush=True)
    train_features = vectorizer.fit_transform(train["prompt"])
    validation_features = vectorizer.transform(validation["prompt"])
    test_features = vectorizer.transform(test["prompt"])

    all_candidates: dict[str, str | dict[str, float]] = {
        "balanced": "balanced",
        "sqrt_balanced": sqrt_balanced_weights(train["label"]),
    }
    if args.weighting == "auto":
        candidates = all_candidates
    else:
        candidates = {
            args.weighting: all_candidates[args.weighting]
        }
    candidate_results: dict[str, Any] = {}
    fitted_candidates: dict[str, LogisticRegression] = {}

    for candidate_name, class_weight in candidates.items():
        print(f"Training candidate: {candidate_name}...", flush=True)
        model = make_classifier(
            class_weight,
            c_value=args.c_value,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            solver=args.solver,
        )
        model.fit(train_features, train["label"])
        validation_predictions = model.predict(validation_features).tolist()
        candidate_results[candidate_name] = {
            "class_weight": class_weight,
            **evaluate_predictions(
                validation["label"],
                validation_predictions,
            ),
            "iterations": [int(value) for value in model.n_iter_],
        }
        fitted_candidates[candidate_name] = model

    selected_name = max(
        candidate_results,
        key=lambda name: (
            candidate_results[name]["macro_f1"],
            candidate_results[name]["weighted_f1"],
        ),
    )
    selected_model = fitted_candidates[selected_name]
    print(f"Selected candidate: {selected_name}", flush=True)

    validation_predictions, validation_confidence, validation_margins = (
        probability_metrics(selected_model, validation_features)
    )
    test_predictions, test_confidence, test_margins = probability_metrics(
        selected_model,
        test_features,
    )

    validation_metrics = {
        **evaluate_predictions(
            validation["label"],
            validation_predictions,
        ),
        "by_source": evaluate_by_source(
            validation,
            validation_predictions,
        ),
    }
    test_metrics = {
        **evaluate_predictions(test["label"], test_predictions),
        "by_source": evaluate_by_source(test, test_predictions),
    }

    save_evaluation_files(
        output_dir=output_dir,
        split_name="validation",
        frame=validation,
        predictions=validation_predictions,
        confidence=validation_confidence,
        margins=validation_margins,
        metrics=validation_metrics,
    )
    save_evaluation_files(
        output_dir=output_dir,
        split_name="test",
        frame=test,
        predictions=test_predictions,
        confidence=test_confidence,
        margins=test_margins,
        metrics=test_metrics,
    )

    review_frames: list[pd.DataFrame] = []
    for split_name, frame, predictions, confidence in (
        (
            "validation",
            validation,
            validation_predictions,
            validation_confidence,
        ),
        ("test", test, test_predictions, test_confidence),
    ):
        review = frame.copy()
        review["split"] = split_name
        review["prediction"] = predictions
        review["confidence"] = confidence
        review = review[
            review["label"].ne(review["prediction"])
            & review["confidence"].ge(0.90)
        ]
        review_frames.append(review)
    probable_label_issues = pd.concat(
        review_frames,
        ignore_index=True,
    ).sort_values("confidence", ascending=False)
    probable_label_issues.to_csv(
        output_dir / "probable_label_issues.csv",
        index=False,
    )

    save_top_features(
        vectorizer=vectorizer,
        model=selected_model,
        output_path=output_dir / "top_features.csv",
    )

    model_bundle = {
        "vectorizer": vectorizer,
        "classifier": selected_model,
        "labels": LABELS,
        "selected_candidate": selected_name,
        "model_version": "1.0.0",
    }
    joblib.dump(
        model_bundle,
        output_dir / "promptshield_tfidf_logreg.joblib",
        compress=3,
    )

    metadata = {
        "model": "TF-IDF + Logistic Regression",
        "selected_candidate": selected_name,
        "candidate_validation_results": candidate_results,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "train_matrix_shape": list(train_features.shape),
            "train_matrix_nonzero_values": int(train_features.nnz),
            "max_features": args.max_features,
            "max_iterations": args.max_iterations,
            "c_value": args.c_value,
            "tolerance": args.tolerance,
            "solver": args.solver,
            "converged": bool(
                all(
                    int(value) < args.max_iterations
                    for value in selected_model.n_iter_
                )
            ),
            "probable_label_issue_rows": len(probable_label_issues),
            "elapsed_seconds": round(time.time() - started_at, 2),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    readme = [
        "# PromptShield Model Report",
        "",
        "Model: TF-IDF word unigrams/bigrams with Logistic Regression.",
        "",
        f"Selected class weighting: `{selected_name}`",
        f"Vocabulary size: {len(vectorizer.vocabulary_):,}",
        f"Training rows: {len(train):,}",
        "",
        "## Validation",
        "",
        f"- Accuracy: {validation_metrics['accuracy']:.4f}",
        f"- Macro F1: {validation_metrics['macro_f1']:.4f}",
        f"- Weighted F1: {validation_metrics['weighted_f1']:.4f}",
        "",
        "## Test",
        "",
        f"- Accuracy: {test_metrics['accuracy']:.4f}",
        f"- Macro F1: {test_metrics['macro_f1']:.4f}",
        f"- Weighted F1: {test_metrics['weighted_f1']:.4f}",
        "",
        "## Test Per-Class F1",
        "",
    ]
    for label in LABELS:
        readme.append(
            f"- {label}: "
            f"{test_metrics['per_class'][label]['f1']:.4f} "
            f"(support: {test_metrics['per_class'][label]['support']:,})"
        )
    readme.extend(
        [
            "",
            "## Data Quality Note",
            "",
            (
                f"- {len(probable_label_issues):,} validation/test rows have "
                "a different prediction with at least 90% confidence and "
                "are exported for manual label review."
            ),
            (
                "- Prompt Injection and Data Extraction have small test "
                "support, so their individual scores are less stable than "
                "the Safe, Jailbreak, and Roleplay scores."
            ),
            "",
        "Per-class metrics, confusion matrices, predictions, and the "
        "highest-confidence errors are stored in the validation and test "
        "subfolders.",
        ]
    )
    (output_dir / "README.md").write_text(
        "\n".join(readme) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
