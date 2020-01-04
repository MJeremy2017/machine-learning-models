import numpy as np
import matplotlib.pyplot as plt


def fx(x):
    return x**2


x_range = np.linspace(-1, 1, 100)
y_value = [fx(x) for x in x_range]


def gd(init_x, grad_fn, lr=0.01, n_iter=10):
    x = init_x
    x_list = [x]
    for i in range(n_iter):
        x -= lr*grad_fn(x)
        x_list.append(x)
    return x_list


init_x = -1
grad_fn = lambda x: 2*x

x_list = gd(init_x, grad_fn, lr=0.02, n_iter=100)


def fx2(x):
    return 2*x + 3


x_range = np.linspace(-1, 1, 100)
y_value = [fx2(x) for x in x_range]


def gd2(inits, X, Y, lr=0.01, n_iter=10):
    n = len(X)
    a, b = inits
    grad_a, grad_b = lambda x, y: -2*x*(y-(a*x+b)), lambda x, y: -2*(y-(a*x+b))
    a_list, b_list = [a], [b]
    for i in range(n_iter):
        for j in range(n):
            x_j, y_j = X[j], Y[j]
            a -= lr*grad_a(x_j, y_j)
            b -= lr*grad_b(x_j, y_j)
            a_list.append(a)
            b_list.append(b)
    return a_list, b_list


inits = [0, 0]
a_list, b_list = gd2(inits, x_range, y_value, n_iter=10)


def gd3(inits, X, Y, lr=0.01, n_iter=10):
    n = len(X)
    a, b = inits
    grad_a, grad_b = lambda x, y: -2*x*(y-(a*x+b)), lambda x, y: -2*(y-(a*x+b))
    a_list, b_list = [a], [b]
    for i in range(n_iter):
        grad_sum_a = 0
        grad_sum_b = 0
        for j in range(n):
            x_j, y_j = X[j], Y[j]
            grad_sum_a += grad_a(x_j, y_j)
            grad_sum_b += grad_b(x_j, y_j)
        a -= lr*grad_sum_a/n
        b -= lr*grad_sum_b/n
        a_list.append(a)
        b_list.append(b)
    return a_list, b_list


def plot_gd(a_list, b_list):
    plt.figure(figsize=[8, 4])
    plt.plot(range(len(a_list)), a_list, label="a")
    plt.plot(range(len(b_list)), b_list, label="b")
    plt.xlabel("n_iteration")
    plt.legend()


inits = [0, 0]
a_list, b_list = gd3(inits, x_range, y_value, lr=0.1, n_iter=100)

plot_gd(a_list, b_list)