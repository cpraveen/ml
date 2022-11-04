"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import numpy as np


def func(x):
    """
    x: array_like, N x D_in
    y: array_like, N x D_out
    """
    return 2.0 + np.sin(8 * np.pi * x)


geom = dde.geometry.Interval(0, 1)
num_train = 128
num_test = 128
data = dde.data.Function(geom, func, num_train, num_test, 
                         train_distribution="pseudo")

activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN([1] + [20] * 5 + [1], activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=20000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
