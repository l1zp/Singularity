import numpy as np
from pydiablo import Variable
import pydiablo.functions as F

# build dataset
np.random.seed(0)

x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)

# build model
W = Variable(0.01 * np.random.randn(1, 1))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b
    return y


# train model
lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_square_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)
