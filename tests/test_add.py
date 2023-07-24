import numpy as np
import unittest

from pydiablo.core import Variable
from pydiablo.functions import add, square


class AddTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.))
        y = Variable(np.array(3.))
        z = add(x, y)
        expected = np.array(5.)
        self.assertEqual(z.data, expected)

    def test_backward(self):
        x = Variable(np.array(2.))
        y = Variable(np.array(3.))
        z = add(square(x), square(y))
        z.backward()
        expected = np.array(4.)
        self.assertEqual(x.grad, expected)
