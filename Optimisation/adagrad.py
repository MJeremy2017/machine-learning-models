import numpy as np


def adagrad(inits, X, Y, lr=0.01, n_iter=10, epsilon=1e-6):
    n = len(X)
    a, b = inits
    grad_a, grad_b = lambda x, y: -2 * x * (y - (a * x + b)), lambda x, y: -2 * (y - (a * x + b))
    s_a, s_b = 0, 0
    a_list, b_list = [a], [b]
    a_lr_list, b_lr_list = [], []
    for _ in range(n_iter):
        for i in range(n):
            x_i, y_i = X[i], Y[i]

            s_a += (grad_a(x_i, y_i)) ** 2
            s_b += (grad_b(x_i, y_i)) ** 2

            lr_a = lr / np.sqrt(s_a + epsilon)
            lr_b = lr / np.sqrt(s_b + epsilon)

            a -= grad_a(x_i, y_i) * lr_a
            b -= grad_b(x_i, y_i) * lr_b

            a_lr_list.append(lr_a)
            b_lr_list.append(lr_b)
            a_list.append(a)
            b_list.append(b)
    return a_list, b_list, a_lr_list, b_lr_list


def adagrad_batch(inits, X, Y, lr=0.01, n_iter=10, batch_size=50, epsilon=1e-6, shuffle=True):
    n = len(X)
    ind = list(range(n))
    a, b = inits
    grad_a, grad_b = lambda x, y: -2 * x * (y - (a * x + b)), lambda x, y: -2 * (y - (a * x + b))
    s_a, s_b = 0, 0
    a_list, b_list = [a], [b]
    a_lr_list, b_lr_list = [], []
    for _ in range(n_iter):
        if shuffle:
            np.random.shuffle(ind)  # shuffle the index on every iteration
        batch_indices = [ind[i:(i + batch_size)] for i in range(0, len(ind), batch_size)]
        for indices in batch_indices:
            grad_sum_a = 0
            grad_sum_b = 0
            # each batch compute total gradient
            for j in indices:
                x_j, y_j = X[j], Y[j]
                grad_sum_a += grad_a(x_j, y_j)
                grad_sum_b += grad_b(x_j, y_j)
            # update on average gradient
            grad_avg_a, grad_avg_b = grad_sum_a / batch_size, grad_sum_b / batch_size
            s_a += grad_avg_a ** 2
            s_b += grad_avg_b ** 2

            lr_a = lr / np.sqrt(s_a + epsilon)
            lr_b = lr / np.sqrt(s_b + epsilon)

            a -= grad_avg_a * lr_a
            b -= grad_avg_b * lr_b

            a_lr_list.append(lr_a)
            b_lr_list.append(lr_b)
            a_list.append(a)
            b_list.append(b)
    return a_list, b_list, a_lr_list, b_lr_list
