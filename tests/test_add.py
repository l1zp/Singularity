import numpy as np
import unittest

from pydiablo.core import Function, Variable
from pydiablo.functions import add, square

x = Variable(np.array(2.))
y = Variable(np.array(3.))

z = add(x, x)
z.backward()
print(z.data)
print(x.grad, y.grad)
