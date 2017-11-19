import unittest
from pbase import layer
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class TestLayer(unittest.TestCase):
    def test_MLP(self):
        mlp = layer.MLP(10, 20, F.elu)
        input = Variable(torch.FloatTensor(*(5, 10)))
        self.assertEqual((5,20), mlp(input).size())

    def test_simpleCNN(self):
        input = Variable(torch.FloatTensor(*(5, 10, 300)))
        kimcnn = layer.SimpleCNN(num_of_conv=3, in_channels=1, out_channels=100,
                                 kernel_size=[2, 3, 4], in_features=300, out_features=100)
        x = kimcnn(input)
        self.assertEqual(torch.Size([5, 100]), x.size())

    def test_CharCNN(self):
        input = Variable(torch.FloatTensor(*(5, 10, 4, 50)))
        charcnn = layer.CharCNN(num_of_conv=1, in_channels=1, out_channels=100,
                                kernel_size=2, in_features=50, out_features=300)
        x = charcnn(input)
        self.assertEqual(torch.Size([5, 10, 300]), x.size())

    def test_charEmbedding(self):
        input = Variable(torch.LongTensor([[[1, 3, 4, 5],[4, 1, 3, 0]]]))
        # input = (sent=1, word=2, char=4)
        charEmbedding = layer.CharEmbedding(6, 10)
        x = charEmbedding(input)
        self.assertEqual(torch.Size([1, 2, 4, 10]), x.size())


if __name__=="__main__":
    unittest.main()

