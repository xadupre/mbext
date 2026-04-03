# ModelBuilder for onnxruntime-genai

The code base comes from https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models.
It adds fast unit tests checking discrepancies, end to end test with the trained model.
It supports more architectures.

## Style

```bash
black . && ruff check .
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
pytest tests/trained
```
