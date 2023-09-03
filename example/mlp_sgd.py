import numpy as np
from pydiablo.models import MLP
from pydiablo.optimizers import SGD

import pydiablo.functions as F


# build data set
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# hyper parameters
lr = 0.2
max_iter = 10000
hidden_size = 10

# build model
model = MLP((hidden_size, 1))
optimizer = SGD(lr=lr)
optimizer.setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_square_error(y, y_pred)
    model.cleargrads()
    loss.backward()
    optimizer.update()
    if i % 1000 == 0:
        print(f"iter: {i}, loss: {loss.data:.3f}")
