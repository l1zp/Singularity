import numpy as np

from functions import Square, Exp
from core import Variable, Function

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)

y.grad = np.array(1.0)
y.backward()
print(x.grad)
