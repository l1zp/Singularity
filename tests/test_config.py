import numpy as np
import unittest

from pydiablo import Variable, no_grad
from pydiablo.functions import square


class AddTest(unittest.TestCase):
    def test_forward(self):
        with no_grad():
            x = Variable(np.array(2.0))
            y = square(x)
