import numpy as np


def fx2(x):
    return 2*x + 3


x_range = np.linspace(-1, 1, 100)
y_value = [fx2(x) for x in x_range]


def sgd(inits, X, Y, lr=0.01, n_iter=10):
    n = len(X)
    ind = list(range(n))
    a, b = inits
    grad_a, grad_b = lambda x, y: -2*x*(y-(a*x+b)), lambda x, y: -2*(y-(a*x+b))
    a_list, b_list = [a], [b]
    for i in range(n_iter):
        np.random.shuffle(ind)  # shuffle the index on every iteration
        for j in ind:
            x_j, y_j = X[j], Y[j]
            a -= lr*grad_a(x_j, y_j)
            b -= lr*grad_b(x_j, y_j)
            a_list.append(a)
            b_list.append(b)
    return a_list, b_list


def sgd(inits, X, Y, lr=0.01, n_iter=10, batch_size=50, shuffle=True):
    n = len(X)
    ind = list(range(n))
    a, b = inits
    grad_a, grad_b = lambda x, y: -2*x*(y-(a*x+b)), lambda x, y: -2*(y-(a*x+b))
    a_list, b_list = [a], [b]
    for i in range(n_iter):
        if shuffle:
            np.random.shuffle(ind)  # shuffle the index on every iteration
        batch_indices = [ind[i:(i+batch_size)] for i in range(0, len(ind), batch_size)]
        for indices in batch_indices:
            grad_sum_a = 0
            grad_sum_b = 0
            for j in indices:
                x_j, y_j = X[j], Y[j]
                grad_sum_a += grad_a(x_j, y_j)
                grad_sum_b += grad_b(x_j, y_j)
            a -= lr*grad_sum_a/batch_size
            b -= lr*grad_sum_b/batch_size
            a_list.append(a)
            b_list.append(b)
    return a_list, b_list


def sgd_mom(inits, X, Y, lr=0.01, n_iter=10, gamma=0.9):
    n = len(X)
    ind = list(range(n))
    a, b = inits
    grad_a, grad_b = lambda x, y: -2*x*(y-(a*x+b)), lambda x, y: -2*(y-(a*x+b))
    v_a, v_b = 0, 0
    a_list, b_list = [a], [b]
    for i in range(n_iter):
        np.random.shuffle(ind)  # shuffle the index on every iteration
        for j in ind:
            x_j, y_j = X[j], Y[j]
            # update momentum
            v_a = gamma*v_a + lr*grad_a(x_j, y_j)
            v_b = gamma*v_b + lr*grad_b(x_j, y_j)
            # update params
            a -= v_a
            b -= v_b
            a_list.append(a)
            b_list.append(b)
    return a_list, b_list