import numpy as np

from functions import square, exp
from core import Variable, Function

x = Variable(np.array(0.5))
# x = Variable(0.1) raise type error
y = square(exp(square(x)))
print(y.data)

y.backward()
print(x.grad)
