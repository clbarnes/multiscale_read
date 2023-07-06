.PHONY: format
format:
	poetry run ruff check --fix-only . &&\
	poetry run black .

.PHONY: lint-notype
lint-notype:
	poetry run black --check .
	poetry run ruff check .

.PHONY: lint
lint: lint-notype
	poetry run mypy .

.PHONY: test
test:
	poetry run pytest --verbose

.PHONY: install
install:
	poetry install --all-extras

.PHONY: install-dev
install-dev:
	poetry install --all-extras --with dev

.PHONY: clean-docs
clean-docs:
	rm -rf docs

.PHONY: docs
docs: clean-docs
	mkdir -p docs &&\
	poetry run pdoc --html --output-dir docs multiscale_read
