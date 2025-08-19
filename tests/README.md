# Loes Scoring System - Test Suite

Comprehensive test suite for the Loes Scoring medical AI system.

## Test Structure

```
tests/
├── unit/                      # Unit tests for individual components
│   ├── test_metrics.py       # Metrics calculations (SRMSE, MAE)
│   ├── test_data_structures.py # Data structures (CandidateInfoTuple)
│   ├── test_models.py        # Neural network architectures
│   ├── test_dataset.py       # Dataset loading and augmentation
│   ├── test_training.py      # Training utilities and loops
│   └── test_inference.py     # Inference pipeline and predictions
└── README.md                 # This file
```

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py -v

# Run specific test module
python run_tests.py -m test_metrics

# Stop on first failure
python run_tests.py -f

# Generate coverage report
python run_tests.py -c
```

### Using pytest

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_models.py

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest -n auto

# Run only fast tests
pytest -m "not slow"

# Run only GPU tests
pytest -m gpu
```

### Using unittest

```bash
# Run all tests
python -m unittest discover tests

# Run specific test module
python -m unittest tests.unit.test_metrics

# Verbose output
python -m unittest discover tests -v
```

## Test Categories

### Unit Tests

#### 1. **Metrics Tests** (`test_metrics.py`)
- Standardized RMSE calculation
- Perfect prediction scenarios
- Edge cases (zero variance, outliers)
- Numerical stability
- Batch aggregation

#### 2. **Data Structure Tests** (`test_data_structures.py`)
- CandidateInfoTuple initialization
- Sorting and hashing
- Property accessors
- Data validation
- Path extraction utilities

#### 3. **Model Tests** (`test_models.py`)
- ResNet architecture initialization
- AlexNet3D structure verification
- Forward pass shape validation
- Gradient flow testing
- Weight initialization
- Device handling (CPU/GPU)
- Model serialization

#### 4. **Dataset Tests** (`test_dataset.py`)
- LoesScoreDataset initialization
- Subject filtering
- Random/sorted ordering
- NIfTI file loading
- Data augmentation
- DataLoader integration
- Error handling

#### 5. **Training Tests** (`test_training.py`)
- Training loop components
- Optimizer steps
- Learning rate scheduling
- Gradient accumulation
- Checkpointing
- Early stopping
- Mixed precision training

#### 6. **Inference Tests** (`test_inference.py`)
- Model loading
- Single/batch inference
- Post-processing
- Confidence estimation
- Error handling
- Optimization techniques

## Test Coverage Goals

| Module | Target Coverage | Current Coverage |
|--------|----------------|------------------|
| Metrics | 95% | - |
| Models | 90% | - |
| Dataset | 90% | - |
| Training | 85% | - |
| Inference | 90% | - |

## Writing New Tests

### Test Template

```python
import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

class TestNewFeature(unittest.TestCase):
    """Test suite for new feature."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize test data
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up resources
        pass
    
    def test_normal_case(self):
        """Test normal operation."""
        # Test implementation
        self.assertTrue(True)
    
    def test_edge_case(self):
        """Test edge cases."""
        # Test edge cases
        self.assertIsNotNone(None)
    
    def test_error_handling(self):
        """Test error handling."""
        with self.assertRaises(ValueError):
            # Code that should raise error
            pass

if __name__ == '__main__':
    unittest.main()
```

### Best Practices

1. **Isolation**: Each test should be independent
2. **Clarity**: Use descriptive test names
3. **Coverage**: Test both success and failure paths
4. **Mocking**: Mock external dependencies
5. **Performance**: Keep tests fast (< 1 second each)
6. **Documentation**: Add docstrings to test methods

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure src is in Python path
   export PYTHONPATH="${PYTHONPATH}:./src"
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov pytest-xdist
   ```

3. **CUDA/GPU Tests**
   ```bash
   # Skip GPU tests if no GPU available
   pytest -m "not gpu"
   ```

4. **Slow Tests**
   ```bash
   # Set timeout for slow tests
   pytest --timeout=60
   ```

## Performance Benchmarks

Expected test execution times:

- Unit tests: < 10 seconds total
- Integration tests: < 60 seconds total
- Full suite: < 2 minutes

## Contributing

When adding new features:

1. Write tests FIRST (TDD approach)
2. Ensure all tests pass
3. Maintain > 80% code coverage
4. Update this README if adding new test categories

## Resources

- [Python unittest documentation](https://docs.python.org/3/library/unittest.html)
- [pytest documentation](https://docs.pytest.org/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)