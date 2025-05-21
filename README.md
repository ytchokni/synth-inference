# synthdid_py

A Python implementation of the Synthetic Difference-in-Differences method.

This package is a port of the R package [synthdid](https://github.com/synth-inference/synthdid) to Python.

## Installation

```
pip install synthdid_py
```

## Usage

```python
from synthdid_py.synthdid import synthdid_estimate
import numpy as np

# Simple example
Y = np.random.normal(size=(10, 10))
N0 = 5  # Number of control units
T0 = 5  # Number of pre-treatment periods
estimate = synthdid_estimate(Y, N0, T0)
print(estimate)
```

## Testing

```
pytest tests/
```