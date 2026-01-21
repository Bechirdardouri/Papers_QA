.PHONY: help install install-dev install-all test lint format typecheck clean build docker-build docker-run docker-stop api docs

PYTHON := python
PIP := pip
PROJECT_NAME := papers-qa
SRC_DIR := src/papers_qa
TEST_DIR := tests

help:
	@echo "Papers QA - Development Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make install-all      Install all dependencies including optional"
	@echo ""
	@echo "Development:"
	@echo "  make test             Run tests with coverage"
	@echo "  make test-fast        Run tests without coverage"
	@echo "  make lint             Run linting checks"
	@echo "  make format           Format code with black and ruff"
	@echo "  make typecheck        Run type checking with mypy"
	@echo "  make check            Run all checks (lint, typecheck, test)"
	@echo ""
	@echo "Build & Deploy:"
	@echo "  make build            Build Python package"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run Docker container"
	@echo "  make docker-stop      Stop Docker container"
	@echo "  make docker-compose   Start all services with docker-compose"
	@echo ""
	@echo "Application:"
	@echo "  make api              Start the API server"
	@echo "  make cli-help         Show CLI help"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove build artifacts"
	@echo "  make clean-all        Remove all generated files"
	@echo "  make docs             Generate documentation"

# Installation targets
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

install-all:
	$(PIP) install -e ".[dev,api,docs]"
	pre-commit install

# Testing targets
test:
	pytest $(TEST_DIR)/ -v --cov=$(PROJECT_NAME) --cov-report=term-missing --cov-report=html

test-fast:
	pytest $(TEST_DIR)/ -v

test-integration:
	pytest $(TEST_DIR)/ -v -m integration

# Code quality targets
lint:
	ruff check $(SRC_DIR)/ $(TEST_DIR)/
	ruff format --check $(SRC_DIR)/ $(TEST_DIR)/

format:
	ruff format $(SRC_DIR)/ $(TEST_DIR)/
	ruff check --fix $(SRC_DIR)/ $(TEST_DIR)/

typecheck:
	mypy $(SRC_DIR)/ --ignore-missing-imports

check: lint typecheck test

# Build targets
build:
	$(PYTHON) -m build

# Docker targets
docker-build:
	docker build -t $(PROJECT_NAME):latest .

docker-run:
	docker run -d --name $(PROJECT_NAME) \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		-p 8000:8000 \
		$(PROJECT_NAME):latest

docker-stop:
	docker stop $(PROJECT_NAME) || true
	docker rm $(PROJECT_NAME) || true

docker-compose:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Application targets
api:
	$(PYTHON) -m papers_qa.api

cli-help:
	$(PYTHON) -m papers_qa.cli --help

# Documentation targets
docs:
	cd docs && make html

# Cleanup targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf data/cache/*
	rm -rf data/generated/*
	rm -rf logs/*

# Development utilities
seed-data:
	@echo "Seeding sample data..."
	$(PYTHON) -c "from papers_qa import get_settings; s = get_settings(); print(f'Data dirs created: {s.data.input_dir}, {s.data.output_dir}, {s.data.cache_dir}')"

shell:
	$(PYTHON) -c "from papers_qa import *; import IPython; IPython.embed()"
