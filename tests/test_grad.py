import numpy as np
import unittest

from pydiablo import Variable, no_grad
from pydiablo.functions import square


class GradTest(unittest.TestCase):
    @staticmethod
    def f(x):
        y = x ** 4 - 2 * x ** 2
        return y

    def test_forward(self):
        x = Variable(np.array(2.0))
        y = self.f(x)
        y.backward(create_graph=True)
        self.assertEqual(x.grad.data, np.array(24.0))

        gx = x.grad
        gx.backward()
        self.assertEqual(x.grad.data, np.array(68.0))
