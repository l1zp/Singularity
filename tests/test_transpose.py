import numpy as np
import unittest

from pydiablo import Variable
import pydiablo.functions as F


class ReshapeTest(unittest.TestCase):
    def test(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.transpose(x)
        y.backward()
        self.assertTrue(np.array_equal(x.grad.data, np.array([[1, 1, 1], [1, 1, 1]])))

    def test_var(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertTrue(np.array_equal(x.T.data, np.array([[1, 2, 3], [4, 5, 6]]).T))
