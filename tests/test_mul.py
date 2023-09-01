import numpy as np
import unittest

from pydiablo import Variable
import pydiablo.functions as F


class MulTest(unittest.TestCase):
    def test_forward(self):
        a = np.array(3.0)
        b = Variable(np.array(2.0))
        c = np.array(1.0)
        y = a * b + c
        expected = np.array(7.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(1.0))
        y = a * b + c
        y.backward()
        expected_a = np.array(2.0)
        expected_b = np.array(3.0)
        self.assertEqual(a.grad.data, expected_a)
        self.assertEqual(b.grad.data, expected_b)

    def test_matmul(self):
        x = Variable(np.random.randn(2, 3))
        w = Variable(np.random.randn(3, 4))
        y = F.matmul(x, w)
        y.backward()