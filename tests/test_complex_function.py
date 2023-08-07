import numpy as np
import unittest

from pydiablo import Variable
from pydiablo.utils import plot_dot_graph


class SphereTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.x = Variable(np.array(3.0))
        self.y = Variable(np.array(4.0))

    def sphere(self, x, y):
        z = x**2 + y**2
        return z

    def test_forward(self):
        z = self.sphere(self.x, self.y)
        expected = np.array(25.0)
        self.assertEqual(z.data, expected)

    def test_backward(self):
        z = self.sphere(self.x, self.y)
        z.backward()
        expected_x = np.array(6.0)
        expected_y = np.array(8.0)
        self.assertEqual(self.x.grad.data, expected_x)
        self.assertEqual(self.y.grad.data, expected_y)


class MatyasTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.x = Variable(np.array(1.0))
        self.y = Variable(np.array(1.0))

    def matyas(self, x, y):
        z = 0.26 * (x**2 + y**2) - 0.48 * x * y
        return z

    def test_forward(self):
        z = self.matyas(self.x, self.y)
        expected = np.array(0.040000000000000036)
        self.assertEqual(z.data, expected)

    def test_backward(self):
        z = self.matyas(self.x, self.y)
        z.backward()
        expected_x = np.array(0.040000000000000036)
        expected_y = np.array(0.040000000000000036)
        self.assertEqual(self.x.grad.data, expected_x)
        self.assertEqual(self.y.grad.data, expected_y)


class GPTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.x = Variable(np.array(1.0))
        self.x.name = "x"
        self.y = Variable(np.array(1.0))
        self.y.name = "y"

    def goldstein(self, x, y):
        z = (
            1
            + (x + y + 1) ** 2
            * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
        ) * (
            30
            + (2 * x - 3 * y) ** 2
            * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
        )
        return z

    def test_forward(self):
        z = self.goldstein(self.x, self.y)
        expected = np.array(1876.0)
        self.assertEqual(z.data, expected)

    def test_backward(self):
        z = self.goldstein(self.x, self.y)
        z.backward()
        expected_x = np.array(-5376.0)
        expected_y = np.array(8064.0)
        self.assertEqual(self.x.grad.data, expected_x)
        self.assertEqual(self.y.grad.data, expected_y)

    def test_graph_plot(self):
        z = self.goldstein(self.x, self.y)
        z.name = "z"
        z.backward()
        plot_dot_graph(z, verbose=False)
        

