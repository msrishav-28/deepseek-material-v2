.PHONY: help install dev-install test test-unit test-integration test-scientific test-performance test-reproducibility test-all coverage lint format clean docker-up docker-down init validate-data

help:
	@echo "Ceramic Armor Discovery Framework - Make Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install       - Install package in production mode"
	@echo "  make dev-install   - Install package in development mode"
	@echo "  make init          - Initialize environment"
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run unit tests"
	@echo "  make test-unit         - Run unit tests only"
	@echo "  make test-integration  - Run integration tests"
	@echo "  make test-scientific   - Run scientific accuracy tests"
	@echo "  make test-performance  - Run performance tests"
	@echo "  make test-reproducibility - Run reproducibility tests"
	@echo "  make test-all          - Run all tests including slow ones"
	@echo "  make coverage          - Generate coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Format code with black"
	@echo "  make validate-data - Validate test data"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Clean build artifacts"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v -m "not slow and not performance"

test-unit:
	pytest tests/ -v -m "unit or not (integration or performance or scientific or reproducibility)"

test-integration:
	pytest tests/ -v -m "integration"

test-scientific:
	pytest tests/ -v -m "scientific"

test-performance:
	pytest tests/ -v -m "performance"

test-reproducibility:
	pytest tests/ -v -m "reproducibility"

test-all:
	pytest tests/ -v

coverage:
	pytest tests/ --cov=ceramic_discovery --cov-report=html --cov-report=term
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	mypy src/ceramic_discovery --ignore-missing-imports

format:
	black src/ tests/ --line-length=100

validate-data:
	python tests/utils/validate_test_data.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

init:
	@echo "Initializing Ceramic Armor Discovery Framework..."
	@mkdir -p data/hdf5
	@mkdir -p results
	@mkdir -p logs
	@cp -n .env.example .env || true
	@echo "✓ Directories created"
	@echo "✓ Environment file ready (.env)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env with your API keys"
	@echo "  2. Run 'make docker-up' to start services"
	@echo "  3. Run 'ceramic-discovery init' to validate setup"
