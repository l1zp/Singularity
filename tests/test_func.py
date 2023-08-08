import numpy as np
import unittest

from pydiablo import Variable
from pydiablo.functions import sin


class SinTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(np.pi/4))
        y = sin(x)
        expected = 1 / np.sqrt(2.)
        self.assertAlmostEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(np.pi/4))
        y = sin(x)
        y.backward()
        expected = 1 / np.sqrt(2.)
        self.assertAlmostEqual(x.grad, expected)
