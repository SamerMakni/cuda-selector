# Cuda_selector

A simple tool to select the optimal CUDA device based on memory, power, or utilization.


### Install
```bash
pip install cuda-selector
```

```python
from cuda_selector import auto_cuda

# Select cuda device with most memory available
device = auto_cuda()

# Select cuda device with lowest power usasge usage
device = auto_cuda('power')

# Select cuda device with lowest power usasge usage
device = auto_cuda('utilization')
```