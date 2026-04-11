# Octopus

GPU worker parallelization for model inference. Supports PyTorch, ONNX Runtime, and AIMET QuantSim models.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from octopus import Octopus

model = my_pytorch_model
def eval_fn(model, batch):
    return model(batch)

with Octopus(model=model, eval_fn=eval_fn) as o:
    for result in o(dataset):
        process(result)
```
