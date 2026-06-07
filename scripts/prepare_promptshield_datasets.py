from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd


CORE_FILES = {
    "jailbreak": Path("jailbreak/promptshield_jailbreak_dataset.csv"),
    "injection_benign": Path(
        "prompt injection/promptshield_injection_benign_dataset.csv"
    ),
    "safe_evaluation": Path(
        "prompt-injection-benign-evaluation-framework/"
        "promptshield_safe_prompts.csv"
    ),
}

PHISHING_FILE = Path(
    "phishing email dataset/promptshield_phishing_dataset.csv"
)

ALLOWED_LABELS = {
    "Safe Prompt",
    "Prompt Injection",
    "Jailbreak",
    "Data Extraction",
    "Roleplay Manipulation",
}

SAFE_RELABEL_RULES = (
    (
        re.compile(
            r"^\s*ignore\s+all\s+previous\s+instructions\b",
            flags=re.IGNORECASE,
        ),
        "Prompt Injection",
        "Starts with an explicit instruction-override command.",
    ),
    (
        re.compile(
            r"\bpretend\s+to\s+be\s+DAN\b|\bact\s+as\s+DAN\b",
            flags=re.IGNORECASE,
        ),
        "Roleplay Manipulation",
        "Explicitly requests DAN-style roleplay.",
    ),
)

SPLIT_RANDOM_STATE = 42
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10


def normalize_prompt(value: str) -> str:
    return " ".join(value.split()).casefold()


def load_core_datasets(input_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for source_name, relative_path in CORE_FILES.items():
        path = input_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found: {path}")

        frame = pd.read_csv(path, dtype="string")
        expected_columns = {"prompt", "label"}
        if set(frame.columns) != expected_columns:
            raise ValueError(
                f"{path} must contain exactly {sorted(expected_columns)}; "
                f"found {list(frame.columns)}"
            )

        frame = frame.copy()
        frame["source_dataset"] = source_name
        frame["source_file"] = str(relative_path).replace("\\", "/")
        frame["source_row"] = range(2, len(frame) + 2)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def validate_input(frame: pd.DataFrame) -> None:
    missing_prompt = frame["prompt"].isna() | frame["prompt"].str.strip().eq("")
    missing_label = frame["label"].isna() | frame["label"].str.strip().eq("")
    invalid_labels = sorted(set(frame["label"].dropna()) - ALLOWED_LABELS)

    problems: list[str] = []
    if missing_prompt.any():
        problems.append(f"{int(missing_prompt.sum())} missing/empty prompts")
    if missing_label.any():
        problems.append(f"{int(missing_label.sum())} missing/empty labels")
    if invalid_labels:
        problems.append(f"invalid labels: {invalid_labels}")

    if problems:
        raise ValueError("Input validation failed: " + "; ".join(problems))


def relabel_high_confidence_safe_attacks(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = frame.copy()
    cleaned["original_label"] = cleaned["label"]
    cleaned["relabel_reason"] = pd.NA

    safe_source = cleaned["source_dataset"].eq("safe_evaluation")
    still_safe = cleaned["label"].eq("Safe Prompt")

    for pattern, replacement_label, reason in SAFE_RELABEL_RULES:
        matches = (
            safe_source
            & still_safe
            & cleaned["prompt"].str.contains(pattern, na=False)
        )
        cleaned.loc[matches, "label"] = replacement_label
        cleaned.loc[matches, "relabel_reason"] = reason
        still_safe = cleaned["label"].eq("Safe Prompt")

    relabeled = cleaned[cleaned["relabel_reason"].notna()].copy()
    return cleaned, relabeled


def resolve_duplicates(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = frame.copy()
    working["normalized_prompt"] = working["prompt"].map(normalize_prompt)

    conflict_groups = (
        working.groupby("normalized_prompt")["label"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    conflicts = working[
        working["normalized_prompt"].isin(conflict_groups)
    ].copy()

    threat_labels = set(ALLOWED_LABELS) - {"Safe Prompt"}
    for normalized_prompt in conflict_groups:
        group = working[working["normalized_prompt"].eq(normalized_prompt)]
        labels = set(group["label"])
        malicious_labels = labels & threat_labels

        if len(malicious_labels) != 1 or "Safe Prompt" not in labels:
            raise ValueError(
                "Cannot safely resolve conflicting labels for normalized "
                f"prompt: {normalized_prompt[:120]!r}; labels={sorted(labels)}"
            )

        threat_label = next(iter(malicious_labels))
        safe_rows = (
            working["normalized_prompt"].eq(normalized_prompt)
            & working["label"].eq("Safe Prompt")
        )
        working.loc[safe_rows, "label"] = threat_label
        working.loc[safe_rows, "relabel_reason"] = (
            "Duplicate prompt also appears in a threat dataset; "
            f"resolved to {threat_label}."
        )

    conflict_relabels = working[
        working["normalized_prompt"].isin(conflict_groups)
        & working["original_label"].ne(working["label"])
    ].copy()

    duplicate_mask = working.duplicated(
        subset=["normalized_prompt", "label"], keep="first"
    )
    dropped_duplicates = working[duplicate_mask].copy()
    deduplicated = working[~duplicate_mask].copy()

    unresolved = (
        deduplicated.groupby("normalized_prompt")["label"].nunique() > 1
    )
    if unresolved.any():
        raise ValueError(
            f"{int(unresolved.sum())} conflicting normalized prompts remain"
        )

    deduplicated = deduplicated.drop_duplicates(
        subset=["normalized_prompt"], keep="first"
    )
    return (
        deduplicated,
        dropped_duplicates,
        conflicts,
        conflict_relabels,
    )


def split_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_indexes: list[int] = []
    validation_indexes: list[int] = []
    test_indexes: list[int] = []

    strata = frame.groupby(["label", "source_dataset"], sort=True)
    for (label, source_dataset), group in strata:
        indexes = list(group.index)
        seed_material = (
            f"{SPLIT_RANDOM_STATE}|{label}|{source_dataset}".encode("utf-8")
        )
        stable_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8], "big"
        )
        random.Random(stable_seed).shuffle(indexes)

        if len(indexes) < 3:
            train_indexes.extend(indexes)
            continue

        test_count = max(1, round(len(indexes) * TEST_SIZE))
        validation_count = max(
            1, round(len(indexes) * VALIDATION_SIZE)
        )
        if test_count + validation_count >= len(indexes):
            raise ValueError(
                "A label/source stratum is too small for three-way "
                f"splitting: label={label!r}, source={source_dataset!r}, "
                f"rows={len(indexes)}"
            )

        test_indexes.extend(indexes[:test_count])
        validation_indexes.extend(
            indexes[test_count : test_count + validation_count]
        )
        train_indexes.extend(indexes[test_count + validation_count :])

    train = frame.loc[train_indexes]
    validation = frame.loc[validation_indexes]
    test = frame.loc[test_indexes]

    return (
        train.sort_values(["label", "source_dataset", "source_row"]),
        validation.sort_values(["label", "source_dataset", "source_row"]),
        test.sort_values(["label", "source_dataset", "source_row"]),
    )


def class_distribution(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in frame["label"].value_counts().sort_index().items()
    }


def source_distribution(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(source): int(count)
        for source, count in frame["source_dataset"]
        .value_counts()
        .sort_index()
        .items()
    }


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    output_columns = [
        "prompt",
        "label",
        "source_dataset",
        "source_file",
        "source_row",
        "original_label",
        "relabel_reason",
        "normalized_prompt",
    ]
    frame.loc[:, output_columns].to_csv(path, index=False)


def build_report(
    *,
    input_rows: int,
    cleaned: pd.DataFrame,
    relabeled: pd.DataFrame,
    conflict_relabels: pd.DataFrame,
    duplicate_rows: pd.DataFrame,
    conflicts: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    phishing_rows: int | None,
) -> dict[str, object]:
    split_normalized_prompts = {
        "train": set(train["normalized_prompt"]),
        "validation": set(validation["normalized_prompt"]),
        "test": set(test["normalized_prompt"]),
    }
    split_overlap = {
        "train_validation": len(
            split_normalized_prompts["train"]
            & split_normalized_prompts["validation"]
        ),
        "train_test": len(
            split_normalized_prompts["train"]
            & split_normalized_prompts["test"]
        ),
        "validation_test": len(
            split_normalized_prompts["validation"]
            & split_normalized_prompts["test"]
        ),
    }

    return {
        "policy": {
            "phishing": "Excluded from the core PromptShield dataset.",
            "duplicates": (
                "Collapsed after whitespace and case normalization."
            ),
            "conflicting_safe_threat_labels": (
                "Resolved to the single threat label when an identical "
                "prompt appeared as both safe and malicious."
            ),
            "automatic_relabeling": (
                "Limited to explicit instruction-override commands and "
                "DAN-style roleplay in the safe source."
            ),
            "borderline_records": (
                "Left unchanged when a keyword appeared in benign context."
            ),
        },
        "input_core_rows": input_rows,
        "cleaned_unique_rows": len(cleaned),
        "removed_duplicate_rows": len(duplicate_rows),
        "input_conflicting_rows": len(conflicts),
        "automatically_relabeled_rows_before_conflict_resolution": len(
            relabeled
        ),
        "rows_relabeled_during_conflict_resolution": len(
            conflict_relabels
        ),
        "phishing_rows_excluded": phishing_rows,
        "cleaned_class_distribution": class_distribution(cleaned),
        "cleaned_source_distribution": source_distribution(cleaned),
        "splits": {
            "train": {
                "rows": len(train),
                "classes": class_distribution(train),
                "sources": source_distribution(train),
            },
            "validation": {
                "rows": len(validation),
                "classes": class_distribution(validation),
                "sources": source_distribution(validation),
            },
            "test": {
                "rows": len(test),
                "classes": class_distribution(test),
                "sources": source_distribution(test),
            },
        },
        "normalized_prompt_overlap_between_splits": split_overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare clean PromptShield datasets and splits."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("new datasets"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prepared_datasets"),
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_core = load_core_datasets(input_dir)
    validate_input(raw_core)

    relabeled_core, initial_relabels = relabel_high_confidence_safe_attacks(
        raw_core
    )
    (
        cleaned,
        dropped_duplicates,
        conflicts,
        conflict_relabels,
    ) = resolve_duplicates(relabeled_core)
    cleaned = cleaned.sort_values(
        ["label", "source_dataset", "source_row"]
    ).reset_index(drop=True)

    train, validation, test = split_dataset(cleaned)

    phishing_path = input_dir / PHISHING_FILE
    phishing_rows: int | None = None
    if phishing_path.exists():
        phishing_rows = len(pd.read_csv(phishing_path, usecols=["prompt"]))

    write_csv(cleaned, output_dir / "promptshield_cleaned.csv")
    write_csv(train, output_dir / "train.csv")
    write_csv(validation, output_dir / "validation.csv")
    write_csv(test, output_dir / "test.csv")

    audit_columns = [
        "prompt",
        "original_label",
        "label",
        "relabel_reason",
        "source_dataset",
        "source_file",
        "source_row",
    ]
    retained_changed_rows = cleaned[
        cleaned["relabel_reason"].notna()
        | cleaned["original_label"].ne(cleaned["label"])
    ]
    changed_rows = pd.concat(
        [retained_changed_rows, conflict_relabels],
        ignore_index=True,
    ).drop_duplicates(subset=["source_file", "source_row"], keep="first")
    changed_rows.loc[:, audit_columns].to_csv(
        output_dir / "relabel_audit.csv", index=False
    )

    report = build_report(
        input_rows=len(raw_core),
        cleaned=cleaned,
        relabeled=initial_relabels,
        conflict_relabels=conflict_relabels,
        duplicate_rows=dropped_duplicates,
        conflicts=conflicts,
        train=train,
        validation=validation,
        test=test,
        phishing_rows=phishing_rows,
    )
    (output_dir / "preparation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    summary = [
        "# PromptShield Dataset Preparation Report",
        "",
        f"- Core input rows: {report['input_core_rows']:,}",
        f"- Clean unique rows: {report['cleaned_unique_rows']:,}",
        f"- Duplicate rows removed: {report['removed_duplicate_rows']:,}",
        (
            "- High-confidence safe-source rows automatically relabeled: "
            f"{report['automatically_relabeled_rows_before_conflict_resolution']:,}"
        ),
        (
            "- Safe/threat conflict rows relabeled: "
            f"{report['rows_relabeled_during_conflict_resolution']:,}"
        ),
        f"- Phishing rows excluded: {report['phishing_rows_excluded']:,}",
        "",
        "## Clean Class Distribution",
        "",
    ]
    for label, count in report["cleaned_class_distribution"].items():
        summary.append(f"- {label}: {count:,}")

    summary.extend(
        [
            "",
            "## Split Sizes",
            "",
            f"- Train: {report['splits']['train']['rows']:,}",
            f"- Validation: {report['splits']['validation']['rows']:,}",
            f"- Test: {report['splits']['test']['rows']:,}",
            "",
            "Normalized prompt overlap between all splits is zero.",
            "",
            "The phishing dataset remains available for future scope only.",
        ]
    )
    (output_dir / "README.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
