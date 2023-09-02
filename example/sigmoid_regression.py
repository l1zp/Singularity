import numpy as np
from pydiablo import Variable
import pydiablo.functions as F

# build data set
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# initialize parameters
W1 = Variable(0.01 * np.random.randn(1, 10))
b1 = Variable(np.zeros(10))
W2 = Variable(0.01 * np.random.randn(10, 1))
b2 = Variable(np.zeros(1))


# build model
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


# train model
lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_square_error(y, y_pred)
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 1000 == 0:
        print(f"iter: {i}, loss: {loss.data:.3f}")
