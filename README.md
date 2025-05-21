# synthdid_py

Work in Progress of Python implementation of the Synthetic Difference-in-Differences (SDID) method for causal inference.

## Overview

This package is a Python port of the R package [synthdid](https://github.com/synth-inference/synthdid) developed by the original authors of the SDID method. The Synthetic Difference-in-Differences estimator combines the synthetic control method with difference-in-differences to estimate causal effects in panel data settings with multiple treated units and time periods.

SDID is particularly useful when:
- You have panel data with some units receiving treatment and others not
- The treatment occurs at a specific time point
- Parallel trends assumption may be violated
- You need to estimate average treatment effects on the treated

## Source Material

This implementation is based on the paper:

> Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). Synthetic difference-in-differences. *American Economic Review*, 111(12), 4088-4118.

The original paper and R implementation can be found at: [https://github.com/synth-inference/synthdid](https://github.com/synth-inference/synthdid)

## Installation

```
pip install synthdid_py
```

## Usage

Example using California smoking data (as in the original paper):

```python
from synthdid_py import synthdid_estimate, plot_synthdid
import pandas as pd

# Load California smoking data
# Code to load data...

# Estimate treatment effect
estimate = synthdid_estimate(Y, N0, T0)

# Visualize the results
plot_synthdid(estimate)
```

## Examples

For detailed examples, see the examples directory:
- `examples/california_plot_example.py`: Demonstrates how to replicate the California smoking analysis from the original paper

## Testing

```
pytest tests/
```

## Citation

If you use this package in your research, please cite the original paper:

```
@article{arkhangelsky2021synthetic,
  title={Synthetic difference-in-differences},
  author={Arkhangelsky, Dmitry and Athey, Susan and Hirshberg, David A and Imbens, Guido W and Wager, Stefan},
  journal={American Economic Review},
  volume={111},
  number={12},
  pages={4088--4118},
  year={2021}
}
```