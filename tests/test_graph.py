import numpy as np
import unittest

from pydiablo.core import Variable
from pydiablo.functions import square


class AddTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = square(a) + square(a)
        expected = np.array(32.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = square(a) + square(a)
        y.backward()
        expected = np.array(64.0)
        self.assertEqual(x.grad, expected)
