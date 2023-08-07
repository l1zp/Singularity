import numpy as np
import unittest

from pydiablo.core import Variable


class AddTest(unittest.TestCase):
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
        self.assertEqual(a.grad, expected_a)
        self.assertEqual(b.grad, expected_b)
