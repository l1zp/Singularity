import numpy as np

from pydiablo.core import Function, as_variable


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


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x = self.inputs[0].data
        return gy * np.cos(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        (x,) = self.inputs
        return -gy * sin(x)


class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * (1 - y**2)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


class Transpose(Function):
    def forward(self, x):
        return np.transpose(x)

    def backward(self, gy):
        return transpose(gy)


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def sin(x):
    return Sin()(x)


def cos(x):
    return Cos()(x)


def tanh(x):
    return Tanh()(x)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


def transpose(x):
    return Transpose()(x)
