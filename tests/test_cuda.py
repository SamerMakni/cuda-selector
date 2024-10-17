import unittest
from cuda_selector import auto_cuda

class TestAutoCUDA(unittest.TestCase):
    def test_auto_cuda(self):
        result = auto_cuda(criteria='memory')
        self.assertTrue(isinstance(result, str))

if __name__ == '__main__':
    unittest.main()
