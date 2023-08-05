import numpy as np
import unittest

from pydiablo.core import Variable


class AddTest(unittest.TestCase):
    def test_backward(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        t = x0 + x1
        y = t + x0
        y.backward()
        self.assertEqual(y.grad, None)
        self.assertEqual(x0.grad, np.array(2.0))
