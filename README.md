# Named Entity Recognition & Content Analysis

A polished NLP project for exploring fake-vs-real news classification using title and URL text from a public benchmark dataset. This repository brings together an exploratory notebook, a clean Python pipeline, and professional project documentation to make the work easier to follow, extend, and showcase.

## Overview

This project focuses on understanding how language patterns, entity-aware preprocessing, and feature engineering can help distinguish real and fake news content. The workflow includes:

- loading and combining multiple CSV datasets
- cleaning and normalizing text content
- applying spaCy-based lemmatization and stop-word removal
- building TF-IDF features for classification
- evaluating a lightweight model and summarizing its performance

## Highlights

- Clean, modular Python implementation
- Reusable training pipeline in `src/ner_pipeline.py`
- Original research notebook preserved as `Task-1.ipynb`
- Improved project documentation for easier onboarding and portfolio presentation

## Repository structure

```text
.
├── Task-1.ipynb               # Original exploratory notebook
├── src/
│   ├── __init__.py            # Package marker
│   └── ner_pipeline.py        # Reusable NLP data-loading and training pipeline
├── ABOUT.md                   # Project background and scope
├── CONTRIBUTING.md             # Contribution guidance
├── CHANGELOG.md               # Version history
├── README.md                  # Project overview and onboarding
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Python package metadata
├── data/                      # Place dataset CSV files here
└── .gitignore
```

## Data requirements

Place the following dataset files in the `data/` directory before running the pipeline:

- `gossipcop_fake.csv`
- `gossipcop_real.csv`
- `politifact_fake.csv`
- `politifact_real.csv`

## Quick start

```bash
python -m venv .venv
# Windows PowerShell
./.venv/Scripts/Activate.ps1
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m src.ner_pipeline --data-dir data
```

## Tech stack

- Python 3.10+
- pandas
- NumPy
- NLTK
- spaCy
- scikit-learn
- Matplotlib

## Notes

This repository is intentionally structured to keep the exploratory notebook intact while moving reusable logic into a cleaner pipeline for practical use and future experimentation.

## License

This project is currently provided as a research-focused codebase for learning and experimentation. Add a formal license if you plan to share it publicly or use it in a production environment.
