setup:
	python3 -m venv venv/ && . venv/bin/activate
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install