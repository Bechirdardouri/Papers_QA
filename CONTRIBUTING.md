# Contributing to Papers QA

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the Papers QA project.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions. We are committed to providing a welcoming environment for everyone.

## How to Contribute

### 1. Report Issues

Found a bug? Have a feature request? Please open an issue with:
- Clear description of the problem
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Python version and environment

### 2. Submit Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes following our code standards
4. Add tests for new functionality
5. Run tests: `pytest tests/`
6. Commit with clear messages: `git commit -m "Add feature: description"`
7. Push and create a Pull Request

### 3. Code Standards

#### Style Guide
- Use [Black](https://black.readthedocs.io/) for formatting
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Follow [PEP 8](https://pep8.org/) conventions
- Use type hints for all functions

#### Example Function

```python
from typing import Optional, List

def process_documents(
    documents: List[str],
    max_length: Optional[int] = None,
) -> List[dict[str, str]]:
    """Process documents for QA generation.
    
    Args:
        documents: List of document texts.
        max_length: Maximum document length in characters.
    
    Returns:
        List of processed document dictionaries.
    
    Raises:
        ValueError: If documents list is empty.
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")
    
    processed = []
    for doc in documents:
        if max_length and len(doc) > max_length:
            doc = doc[:max_length]
        processed.append({"text": doc})
    
    return processed
```

### 4. Testing

- Write unit tests for new functionality
- Test both success and error cases
- Maintain >80% code coverage
- Run tests locally before submitting PR

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=papers_qa --cov-report=html

# Specific test
pytest tests/test_core.py::TestConfig -v
```

### 5. Documentation

- Update docstrings in code
- Update relevant markdown files
- Add examples for new features
- Keep API documentation current

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/papers-qa.git
cd papers-qa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

## Pre-commit Hooks

We use pre-commit to maintain code quality. Hooks run automatically on commit:

```bash
# Run manually
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

## Project Structure

```
papers_qa/
├── src/papers_qa/      # Main package
│   ├── config.py       # Configuration
│   ├── logging_config.py  # Logging
│   ├── data/           # Data loading
│   ├── retrieval/      # Vector search
│   ├── llm/            # LLM inference
│   ├── generation/     # QA generation
│   └── evaluation/     # Metrics
├── tests/              # Test suite
├── notebooks/          # Example notebooks
├── pyproject.toml      # Project metadata
└── requirements.txt    # Dependencies
```

## Commit Message Guidelines

- Use clear, descriptive messages
- Start with a verb: "Add", "Fix", "Update", "Refactor"
- Reference issues: "Fix #123"
- Examples:
  - `Add support for custom embedding models`
  - `Fix memory leak in batch processor`
  - `Update documentation for API`
  - `Refactor retrieval module for performance`

## Pull Request Guidelines

1. **Title**: Clear and descriptive
2. **Description**: Explain changes and motivation
3. **Tests**: Include tests for new functionality
4. **Documentation**: Update relevant docs
5. **Breaking Changes**: Clearly mark if any

### PR Checklist

- [ ] Code follows style guide
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Passes CI/CD

## Review Process

1. PRs are reviewed by maintainers
2. Feedback may be requested
3. Tests must pass
4. At least one approval required
5. Merge when ready

## Release Process

1. Version bumped in `pyproject.toml`
2. Changes documented in `CHANGELOG.md`
3. Tag created: `git tag v1.0.0`
4. Release published to PyPI
5. GitHub release created

## Areas for Contribution

- [ ] New embedding models
- [ ] Improved LLM support
- [ ] Additional evaluation metrics
- [ ] Performance optimizations
- [ ] Documentation improvements
- [ ] Test coverage expansion
- [ ] Bug fixes
- [ ] Docker optimizations

## Questions?

- Check [Documentation](./SETUP_GUIDE.md)
- Open a [Discussion](https://github.com/yourusername/papers-qa/discussions)
- Email: support@papersqa.com

Thank you for contributing to Papers QA.
