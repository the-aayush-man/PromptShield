# PromptShield Model Report

Model: TF-IDF word unigrams/bigrams with Logistic Regression.

Selected class weighting: `sqrt_balanced`
Vocabulary size: 60,000
Training rows: 142,576

## Validation

- Accuracy: 0.9843
- Macro F1: 0.9476
- Weighted F1: 0.9843

## Test

- Accuracy: 0.9851
- Macro F1: 0.9682
- Weighted F1: 0.9851

## Test Per-Class F1

- Safe Prompt: 0.9909 (support: 13,762)
- Prompt Injection: 0.9767 (support: 22)
- Jailbreak: 0.9457 (support: 2,401)
- Data Extraction: 0.9333 (support: 16)
- Roleplay Manipulation: 0.9941 (support: 1,623)

## Data Quality Note

- 55 validation/test rows have a different prediction with at least 90% confidence and are exported for manual label review.
- Prompt Injection and Data Extraction have small test support, so their individual scores are less stable than the Safe, Jailbreak, and Roleplay scores.

Per-class metrics, confusion matrices, predictions, and the highest-confidence errors are stored in the validation and test subfolders.
