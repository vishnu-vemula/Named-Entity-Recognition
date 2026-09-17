# Contributing

Thank you for your interest in improving this project.

## How to contribute

1. Fork the repository and create a feature branch.
2. Keep changes focused and well-documented.
3. Validate the code locally before submitting a pull request.
4. Document any new dependencies or workflow changes in the README.

## Development workflow

```bash
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m src.ner_pipeline --data-dir data
```

## Code style

- Prefer small, readable functions.
- Add clear docstrings when introducing new functionality.
- Keep notebook experiments separate from reusable pipeline code.

## Pull request guidance

- Explain the problem being solved.
- Include a short summary of the validation performed.
- Keep the scope narrow and easy to review.
