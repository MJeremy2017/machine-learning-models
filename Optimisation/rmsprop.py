import numpy as np


def rmsprop(inits, X, Y, lr=0.01, n_iter=10, gamma=0.9, epsilon=1e-6):
    n = len(X)
    a, b = inits
    grad_a, grad_b = lambda x, y: -2 * x * (y - (a * x + b)), lambda x, y: -2 * (y - (a * x + b))
    s_a, s_b = 0, 0
    a_list, b_list = [a], [b]
    a_lr_list, b_lr_list = [], []
    for _ in range(n_iter):
        for i in range(n):
            x_i, y_i = X[i], Y[i]

            s_a = gamma * s_a + (1 - gamma) * (grad_a(x_i, y_i)) ** 2
            s_b = gamma * s_b + (1 - gamma) * (grad_b(x_i, y_i)) ** 2

            lr_a = lr / np.sqrt(s_a + epsilon)
            lr_b = lr / np.sqrt(s_b + epsilon)

            a -= grad_a(x_i, y_i) * lr_a
            b -= grad_b(x_i, y_i) * lr_b

            a_lr_list.append(lr_a)
            b_lr_list.append(lr_b)
            a_list.append(a)
            b_list.append(b)
    return a_list, b_list, a_lr_list, b_lr_list
