from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rule_engine import scan_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / ".python_packages"
MODEL_PATH = PROJECT_ROOT / "model_artifacts" / "promptshield_tfidf_logreg.joblib"
METADATA_PATH = PROJECT_ROOT / "model_artifacts" / "model_metadata.json"

if PACKAGE_DIR.exists():
    sys.path.insert(0, str(PACKAGE_DIR))

import joblib  # noqa: E402


LABELS = [
    "Safe Prompt",
    "Prompt Injection",
    "Jailbreak",
    "Data Extraction",
    "Roleplay Manipulation",
]


MITIGATIONS = {
    "Safe Prompt": [
        "Allow the prompt to continue to the LLM.",
        "Keep standard logging and abuse monitoring enabled.",
    ],
    "Prompt Injection": [
        "Reject attempts to override system or developer instructions.",
        "Treat all user instructions as untrusted content.",
        "Keep tool access behind explicit policy checks.",
    ],
    "Jailbreak": [
        "Refuse persona or jailbreak instructions.",
        "Restate that safety and system policies remain active.",
        "Route repeated jailbreak attempts for security review.",
    ],
    "Data Extraction": [
        "Do not reveal hidden prompts, credentials, or confidential data.",
        "Mask sensitive fields and enforce least-privilege access.",
        "Log the request for audit and incident review.",
    ],
    "Roleplay Manipulation": [
        "Allow benign roleplay only when safety rules remain unchanged.",
        "Reject role instructions that remove limits or change identity.",
        "Prefer a safe alternative response.",
    ],
}


@dataclass
class ModelContext:
    vectorizer: Any
    classifier: Any
    labels: list[str]
    metadata: dict[str, Any]


_MODEL_CONTEXT: ModelContext | None = None


def load_model() -> ModelContext:
    global _MODEL_CONTEXT
    if _MODEL_CONTEXT is not None:
        return _MODEL_CONTEXT

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    _MODEL_CONTEXT = ModelContext(
        vectorizer=bundle["vectorizer"],
        classifier=bundle["classifier"],
        labels=list(bundle.get("labels", LABELS)),
        metadata=metadata,
    )
    return _MODEL_CONTEXT


def risk_level(score: int) -> str:
    if score >= 85:
        return "Critical"
    if score >= 65:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def severity_level(score: int, label: str) -> str:
    if label == "Safe Prompt" and score < 35:
        return "Benign"
    if score >= 85:
        return "Severe"
    if score >= 65:
        return "High"
    if score >= 35:
        return "Moderate"
    return "Low"


def category_rule_scores(indicators: list[dict[str, object]]) -> dict[str, int]:
    scores = {label: 0 for label in LABELS}
    for indicator in indicators:
        category = str(indicator["category"])
        scores[category] = max(scores[category], int(indicator["weight"]))
    return scores


def explain_result(
    *,
    classification: str,
    ml_label: str,
    ml_confidence: float,
    indicators: list[dict[str, object]],
    risk_score: int,
) -> list[str]:
    explanations: list[str] = []
    if indicators:
        strongest = sorted(
            indicators,
            key=lambda item: int(item["weight"]),
            reverse=True,
        )[:3]
        for indicator in strongest:
            explanations.append(str(indicator["explanation"]))
    else:
        explanations.append("No high-confidence rule indicator was matched.")

    explanations.append(
        "The machine learning model predicted "
        f"{ml_label} with {ml_confidence:.1%} confidence."
    )
    if classification != ml_label:
        explanations.append(
            "The hybrid decision adjusted the model prediction because "
            "rule indicators changed the risk profile."
        )
    if risk_score >= 65:
        explanations.append("The final score is high enough to block or review.")
    elif risk_score >= 35:
        explanations.append("The final score suggests manual review.")
    else:
        explanations.append("The final score is low and the prompt appears safe.")
    return explanations


def analyze_prompt(prompt: str) -> dict[str, Any]:
    text = prompt.strip()
    if not text:
        raise ValueError("Prompt cannot be empty.")
    if len(text) > 20000:
        raise ValueError("Prompt is too long for interactive analysis.")

    context = load_model()
    features = context.vectorizer.transform([text])
    probabilities = context.classifier.predict_proba(features)[0]
    ml_scores = {
        str(label): float(probability)
        for label, probability in zip(context.classifier.classes_, probabilities)
    }
    ml_label = max(ml_scores, key=ml_scores.get)
    ml_confidence = ml_scores[ml_label]

    indicators = scan_prompt(text)
    rule_scores = category_rule_scores(indicators)
    strongest_rule_score = max(rule_scores.values())

    hybrid_scores: dict[str, float] = {}
    for label in LABELS:
        base = ml_scores.get(label, 0.0) * 100
        if label != "Safe Prompt":
            rule_boost = rule_scores[label]
            hybrid_scores[label] = min(100.0, max(base, rule_boost) + rule_boost * 0.18)
        else:
            penalty = strongest_rule_score * 0.55
            hybrid_scores[label] = max(0.0, base - penalty)

    classification = max(hybrid_scores, key=hybrid_scores.get)
    if strongest_rule_score >= 90 and classification == "Safe Prompt":
        classification = max(
            (label for label in LABELS if label != "Safe Prompt"),
            key=lambda label: rule_scores[label],
        )

    confidence = round(hybrid_scores[classification] / 100, 4)
    if classification == "Safe Prompt":
        risk_score = int(round(max(0.0, 100 - hybrid_scores["Safe Prompt"])))
        risk_score = min(risk_score, 34 if not indicators else 64)
    else:
        risk_score = int(
            round(
                min(
                    100.0,
                    hybrid_scores[classification] * 0.72
                    + strongest_rule_score * 0.28,
                )
            )
        )
        risk_score = max(risk_score, strongest_rule_score)

    mitigations = list(MITIGATIONS[classification])
    for indicator in indicators:
        mitigation = str(indicator["mitigation"])
        if mitigation not in mitigations:
            mitigations.append(mitigation)

    result = {
        "classification": classification,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 2),
        "risk_score": risk_score,
        "risk_level": risk_level(risk_score),
        "severity": severity_level(risk_score, classification),
        "ml_prediction": {
            "label": ml_label,
            "confidence": round(ml_confidence, 4),
            "probabilities": {
                label: round(float(ml_scores.get(label, 0.0)), 4)
                for label in LABELS
            },
        },
        "hybrid_scores": {
            label: round(score, 2) for label, score in hybrid_scores.items()
        },
        "rule_score": strongest_rule_score,
        "triggered_indicators": indicators,
        "explanations": explain_result(
            classification=classification,
            ml_label=ml_label,
            ml_confidence=ml_confidence,
            indicators=indicators,
            risk_score=risk_score,
        ),
        "mitigations": mitigations[:6],
    }
    return result


def model_summary() -> dict[str, Any]:
    context = load_model()
    metadata = context.metadata
    return {
        "model": metadata.get("model", "TF-IDF + Logistic Regression"),
        "selected_candidate": metadata.get("selected_candidate", "sqrt_balanced"),
        "test_metrics": metadata.get("test_metrics", {}),
        "training": metadata.get("training", {}),
    }
