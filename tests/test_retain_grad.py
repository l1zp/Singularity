import numpy as np
import unittest

from pydiablo.core import Variable
from pydiablo.functions import add, square


class AddTest(unittest.TestCase):

    def test_backward(self):
        x0 = Variable(np.array(1.))
        x1 = Variable(np.array(1.))
        t = add(x0, x1)
        y = add(t, x0)
        y.backward()
        self.assertEqual(y.grad, None)
        self.assertEqual(x0.grad, np.array(2.))
