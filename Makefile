install_dependencies:
	pip install -r requirements.txt

install_precommit:
	pip install black flake8 isort mypy pre-commit
	pre-commit install --hook-type pre-commit

install: install_dependencies install_precommit