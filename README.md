# ModelBuilder for onnxruntime-genai

The code base comes from https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models.
It adds fast unit tests checking discrepancies, end to end test with the trained model.
It supports more architectures.

## Style

```bash
black . && ruff check .
```

## Development

```bash
pip install -e .[dev]
```

## Fast Unit tests

```bash
pytest tests/fast
```

## Long Unit tests

```bash
python tests/trained/test_trained_tiny_llm.py
```

With a better machine:

```bash
LONGTEST=1 pytest tests/trained
```

You can see the results in ``stats/end2end_results.json``. Example:

```
{'first_diff': 0, 'delta_length': 4, 'expected_length': 16, 'total_diff': 16, 'precision': 'fp32', 'model_id': 'HuggingFaceTB/SmolLM3-3B', 'experiment': 'generate', 'provider': 'cpu'}
{'max_abs_err': 1.6875, '%_gt_0.1': np.float64(0.5278832959081836), '%_gt_0.01': np.float64(0.962624750499002), 'avg_abs_discrepancy': 0.1749267578125, 'shape': (1, 5, 128256), 'dtype': dtype('float16'), 'precision': 'fp16', 'model_id': 'HuggingFaceTB/SmolLM3-3B', 'experiment': 'forward'}
```

