import unittest
from pbase import layer
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class TestLayer(unittest.TestCase):
    """
    Test layer.py
    """
    def test_MLP(self):
        """
        Test MLP
        """
        self.mlp = layer.MLP(10, 20, F.elu)
        input = Variable(torch.FloatTensor(*(5, 10)))
        self.assertEqual((5,20), self.mlp(input).size())

if __name__=="__main__":
    unittest.main()

