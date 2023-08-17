import numpy as np
import unittest

from pydiablo import Variable
from pydiablo.functions import sin, cos, tanh


class SinTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(np.pi/4))
        y = sin(x)
        expected = 1 / np.sqrt(2.)
        self.assertAlmostEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(np.pi/4))
        y = sin(x)
        y.backward()
        expected = 1 / np.sqrt(2.)
        self.assertAlmostEqual(x.grad.data, expected)

class CosTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(np.pi/4))
        y = cos(x)
        self.assertAlmostEqual(y.data, np.cos(np.array(np.pi/4)))

    def test_backward(self):
        x = Variable(np.array(np.pi/4))
        y = cos(x)
        y.backward()
        self.assertAlmostEqual(x.grad.data, -np.sin(np.array(np.pi/4)))

class TanhTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(np.pi/4))
        y = tanh(x)
        self.assertAlmostEqual(y.data, np.tanh(np.array(np.pi/4)))

    def test_backward(self):
        x = Variable(np.array(np.pi/4))
        y = tanh(x)
        y.backward()
        self.assertAlmostEqual(x.grad.data, 1-np.tanh(np.array(np.pi/4))**2)