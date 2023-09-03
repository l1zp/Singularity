import numpy as np
from pydiablo import Variable
import pydiablo.functions as F
import pydiablo.layers as L

# build data set
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# initialize parameters
l1 = L.Linear(10)
l2 = L.Linear(1)


# build model
def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


# train model
lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_square_error(y, y_pred)
    l1.cleargrads()
    l2.cleargrads()
    
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(f"iter: {i}, loss: {loss.data:.3f}")
