import numpy as np
import unittest

from pydiablo import Variable
import pydiablo.functions as F


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
        z = F.square(x) + F.square(y)
        z.backward()
        expected = np.array(4.0)
        self.assertEqual(x.grad.data, expected)

    def test_broadcast(self):
        x = Variable(np.array([1, 2, 3]))
        y = Variable(np.array([10]))
        z = x + y
        expected = np.array([11, 12, 13])
        self.assertTrue(np.array_equal(z.data, expected))

        z.backward()
        self.assertTrue(np.array_equal(y.grad.data, np.array([3])))

    def test_sum(self):
        x = Variable(np.array([1, 2, 3, 4, 5, 6]))
        y = F.sum(x)
        y.backward()
        expected = np.array(21)
        self.assertTrue(np.array_equal(y.data, expected))
        self.assertTrue(np.array_equal(x.grad.data, np.array([1, 1, 1, 1, 1, 1])))

