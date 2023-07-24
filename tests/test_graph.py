import numpy as np
import unittest

from pydiablo.core import Variable
from pydiablo.functions import add, square


class AddTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.))
        a = square(x)
        y = add(square(a), square(a))
        expected = np.array(32.)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(2.))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        expected = np.array(64.)
        self.assertEqual(x.grad, expected)