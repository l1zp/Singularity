import numpy as np
import unittest

from pydiablo.core import Variable
from pydiablo.functions import square


class AddTest(unittest.TestCase):
    def test_forward(self):
        x = np.array([2.0])
        y = Variable(np.array(3.0))
        z = x + y
        expected = np.array(5.0)
        self.assertEqual(z.data, expected)

    def test_backward(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = square(x) + square(y)
        z.backward()
        expected = np.array(4.0)
        self.assertEqual(x.grad, expected)
