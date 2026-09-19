# Convenience targets. Everything here also works as a plain command.

VENV ?= .venv/Scripts/python.exe

.PHONY: test notebooks figures run ablation sensitivity clean

test:
	$(VENV) -m pytest -q

## Generate .ipynb files from the percent-format sources in notebooks/.
notebooks:
	$(VENV) -m jupytext --to notebook notebooks/*.py

## Run every notebook top to bottom and fail if any cell raises.
check-notebooks:
	$(VENV) -m jupytext --to notebook --execute notebooks/0[0-6]*.py

run:
	$(VENV) experiments/run_all.py --config configs/default.yaml

ablation:
	$(VENV) experiments/run_ablation.py

sensitivity:
	$(VENV) experiments/run_sensitivity.py

figures:
	$(VENV) experiments/make_figures.py

clean:
	rm -rf .pytest_cache **/__pycache__ notebooks/*.ipynb
