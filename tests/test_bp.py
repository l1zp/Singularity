import numpy as np
import unittest

from pydiablo.core import Function, Variable
from pydiablo.functions import square, exp


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2))
        # x = Variable(0.1) raise type error
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3))
        # x = Variable(0.1) raise type error
        y = square(x)
        y.backward()
        expected = np.array(6)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        # absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))
        # TODO: check ref 5
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
