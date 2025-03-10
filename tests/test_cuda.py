import unittest
from unittest.mock import patch
import warnings
from cuda_selector import auto_cuda, is_mps_available  

# Mocked nvidia-smi output
MOCK_NVIDIA_SMI_OUTPUT = (
    "38869, 40960, 61.14, 0, 29, 0\n"
    "5174, 40960, 157.02, 50, 33, 1\n"
    "26478, 40960, 299.17, 91, 45, 2\n"
    "39672, 40960, 57.29, 0, 32, 3\n"
    "8394, 40960, 132.15, 44, 42, 4\n"
    "14536, 40960, 246.76, 86, 45, 5\n"
    "26562, 40960, 238.49, 91, 43, 6\n"
    "39308, 40960, 59.75, 0, 32, 7"
)

class TestAutoCuda(unittest.TestCase):
    @patch('subprocess.run')  
    @patch('platform.system', return_value='Linux')  
    @patch('cuda_selector.is_mps_available', return_value=False) 
    def setUp(self, mock_mps, mock_platform, mock_subprocess):
        mock_subprocess.return_value.stdout = MOCK_NVIDIA_SMI_OUTPUT
        self.mock_subprocess = mock_subprocess

    def test_default_memory(self):
        result = auto_cuda()
        self.assertEqual(result, "cuda:3")

    def test_power_criteria(self):
        result = auto_cuda(criteria='power')
        self.assertEqual(result, "cuda:3")

    def test_utilization_criteria(self):
        result = auto_cuda(criteria='utilization')
        self.assertEqual(result, "cuda:3")

    def test_temperature_criteria(self):
        result = auto_cuda(criteria='temperature')
        self.assertEqual(result, "cuda:0")

    def test_multiple_devices(self):
        result = auto_cuda(n=3)
        self.assertEqual(result, ["cuda:3", "cuda:7", "cuda:0"])

    def test_exclude_devices(self):
        result = auto_cuda(exclude=[3])
        self.assertEqual(result, "cuda:7")

    def test_custom_sort_fn(self):
        result = auto_cuda(sort_fn=lambda d: d['memory_free'] * 0.7 + d['utilization'] * 0.3)
        self.assertEqual(result, "cuda:3")

    def test_no_devices_after_filter(self):
        with warnings.catch_warnings(record=True) as w:
            result = auto_cuda(thresholds={'memory_free': 50000})  # No device has > 50000 MB free
            self.assertEqual(result, "cpu")
            self.assertTrue(any("No suitable CUDA devices found" in str(warn.message) for warn in w))

if __name__ == '__main__':
    unittest.main()