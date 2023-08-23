import numpy as np
import unittest

from pydiablo import Variable
import pydiablo.functions as F


class ReshapeTest(unittest.TestCase):
    def test(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.reshape(x, (6,))
        y.backward(retain_grad=True)
        self.assertTrue(np.array_equal(x.grad.data, np.array([[1, 1, 1], [1, 1, 1]])))

    def test_var_reshape(self):
        random_x = np.random.randn(1, 2, 3)
        x = Variable(random_x)
        self.assertTrue(np.array_equal(x.reshape((2, 3)).data, random_x.reshape(2, 3)))
        self.assertTrue(np.array_equal(x.reshape(2, 3).data, random_x.reshape(2, 3)))

