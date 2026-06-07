from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify one prompt with a trained PromptShield model."
    )
    parser.add_argument("prompt")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "model_artifacts/promptshield_tfidf_logreg.joblib"
        ),
    )
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]

    features = vectorizer.transform([args.prompt])
    probabilities = classifier.predict_proba(features)[0]
    ranked = sorted(
        zip(classifier.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )

    result = {
        "classification": ranked[0][0],
        "confidence": round(float(ranked[0][1]), 6),
        "probabilities": {
            label: round(float(probability), 6)
            for label, probability in ranked
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
