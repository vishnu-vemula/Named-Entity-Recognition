# About the Project

This repository explores a lightweight Natural Language Processing workflow for a fake-vs-real news dataset. The original project began as a notebook-driven investigation and has been reorganized into a cleaner, more maintainable structure that is easier to extend, document, and reuse.

## Purpose

The goal is to demonstrate a practical end-to-end pipeline that includes:

- dataset loading and concatenation across multiple sources
- text preprocessing and cleaning
- entity-aware tokenization with spaCy
- feature extraction using TF-IDF
- model training and evaluation with scikit-learn

## Scope

The project is intentionally focused on research and learning. It is designed to help practitioners:

- understand how headline text can be converted into model-ready features
- experiment with data preprocessing choices
- evaluate classification performance on a public fake-news benchmark
- create cleaner project documentation for portfolio or academic use

## Workflow

The repository combines a reusable Python pipeline in `src/ner_pipeline.py` with the original exploratory notebook at `Task-1.ipynb`. This gives an ideal balance between reproducibility and hands-on experimentation.

## Project direction

The design is intentionally simple, professional, and modular so it can evolve over time with new baselines, better preprocessing, or deeper entity analysis without forcing a full rewrite.
