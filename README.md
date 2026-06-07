# PromptShield

PromptShield is a hybrid AI security framework for detecting prompt injection,
jailbreak, data extraction, and roleplay manipulation attacks against large
language model workflows.

## Current Scope

- Local web dashboard
- Rule-based prompt attack detection
- TF-IDF and Logistic Regression ML classifier
- Hybrid risk scoring
- Threat explanations
- Mitigation recommendations
- Prompt history with SQLite
- Analytics charts
- Downloadable PDF reports

## Run

From the project folder:

```powershell
.\start_promptshield.ps1
```

Then open:

```text
http://127.0.0.1:8000
```

## Smoke Test

With the app running, verify the main workflow:

```powershell
$env:PYTHONPATH=(Resolve-Path ".python_packages").Path
& "C:\Users\Aayush\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -B "scripts\smoke_test_promptshield.py"
```

## Important Files

- `server.py` - local web server and API routing
- `app/analyzer.py` - hybrid ML and rule-based analyzer
- `app/rule_engine.py` - prompt security indicators
- `web/index.html` - dashboard
- `model_artifacts/promptshield_tfidf_logreg.joblib` - trained model
- `prepared_datasets/` - cleaned datasets and splits
- `scripts/smoke_test_promptshield.py` - end-to-end workflow check

## Model Summary

- Model: TF-IDF word unigrams/bigrams with Logistic Regression
- Test accuracy: 98.51%
- Test macro F1: 96.82%
- Vocabulary size: 60,000

## Future Scope

- Phishing email detection
- Malicious URL detection
- Real-time monitoring
- Enterprise integration
- AI firewall integrations
