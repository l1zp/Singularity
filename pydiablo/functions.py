import numpy as np

from pydiablo.core import Variable, Function


class Square(Function):
    def forward(self, x):
        return x**2

    def backward(self, gy):
        x = self.inputs[0].data
        return 2 * x * gy


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        return np.exp(x) * gy

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    def backward(self, gy):
        return gy, gy

def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)
