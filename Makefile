.PHONY: format
format:
	ruff check --fix-only . &&\
	black .

.PHONY: lint-notype
lint-notype:
	black --check .
	ruff check .

.PHONY: lint
lint: lint-notype
	mypy .

.PHONY: test
test:
	pytest --verbose

.PHONY: install
install:
	pip install .

.PHONY: install-dev
install-dev:
	pip install -r requirements.txt &&\
	pip install -e .

.PHONY: clean-docs
clean-docs:
	rm -rf docs

.PHONY: docs
docs: clean-docs
	mkdir -p docs &&\
	pdoc --html --output-dir docs multiscale_read
