![PyPI - Version](https://img.shields.io/pypi/v/cuda_selector)
![PyPI - License](https://img.shields.io/pypi/l/cuda_selector)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cuda_selector)
![PyPI - Downloads](https://img.shields.io/pypi/dm/cuda_selector)

# Auto Cuda Selector

A simple tool to select the optimal CUDA device based on memory, power, temperature, or utilization. It supports fallback to CPU and custom sorting functions. Supports CUDA devices on Linux and MPS devices on macOS. Full documentation be found [here](https://samermakni.github.io/cuda-selector/)

### Install
```bash
pip install cuda-selector
```
### Usage
```python
from cuda_selector import auto_cuda

# Select the CUDA device with the most memory available
device = auto_cuda()

# Select the CUDA device with the lowest power usage
device = auto_cuda('power')

# Select the CUDA device with the lowest GPU utilization
device = auto_cuda('utilization')

# Select the CUDA device with the lowest temperature
device = auto_cuda('temperature')

# Select multiple devices with the most free memory
devices = auto_cuda(n=3)

# Exclude specific devices by their index
devices = auto_cuda(exclude=[0, 1])

# Apply thresholds for power usage and utilization
devices = auto_cuda(thresholds={'power': 150, 'utilization': 70})

# Use a custom ranking function for selecting devices
devices = auto_cuda(sort_fn=lambda d: d['memory_free'] * 0.7 + d['utilization'] * 0.3)
```

