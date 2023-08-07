import numpy as np
import unittest

from pydiablo import Variable
from pydiablo.functions import square


class AddTest(unittest.TestCase):
    def test_neg(self):
        x = Variable(np.array(2.0))
        y = -x
        expected = np.array(-2.0)
        self.assertEqual(y.data, expected)

    def test_sub(self):
        x = Variable(np.array(2.0))
        y1 = 1.0 - x
        y2 = x - 1.0
        expected1 = np.array(-1.0)
        expected2 = np.array(1.0)
        self.assertEqual(y1.data, expected1)
        self.assertEqual(y2.data, expected2)

    def test_div(self):
        x = Variable(np.array(2.0))
        y1 = 1.0 / x
        y2 = x / 1.0
        expected1 = np.array(0.5)
        expected2 = np.array(2.0)
        self.assertEqual(y1.data, expected1)
        self.assertEqual(y2.data, expected2)

    def test_pow(self):
        x = Variable(np.array(2.0))
        y = x ** 3
        expected = np.array(8.0)  
        self.assertEqual(y.data, expected)

