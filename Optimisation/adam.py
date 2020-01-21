import numpy as np


def adam(inits, X, Y, lr=0.01, n_iter=10, beta1=0.9, beta2=0.999, epsilon=1e-6):
    n = len(X)
    a, b = inits
    grad_a, grad_b = lambda x, y: -2*x*(y-(a*x+b)), lambda x, y: -2*(y-(a*x+b))
    v_a, v_b = 0, 0
    s_a, s_b = 0, 0
    a_list, b_list = [a], [b]
    t = 1
    for _ in range(n_iter):
        for i in range(n):
            x_i, y_i = X[i], Y[i]
            g_a, g_b = grad_a(x_i, y_i), grad_b(x_i, y_i)
            # compute the first moment
            v_a = beta1*v_a + (1-beta1)*g_a
            v_b = beta1*v_b + (1-beta1)*g_b
            # compute the second moment
            s_a = beta2*s_a + (1-beta2)*(g_a**2)
            s_b = beta2*s_b + (1-beta2)*(g_b**2)
            
            # normalisation
            v_a_norm, v_b_norm = v_a/(1 - np.power(beta1, t)), v_b/(1 - np.power(beta1, t))
            s_a_norm, s_b_norm = s_a/(1 - np.power(beta2, t)), s_b/(1 - np.power(beta2, t))
            t += 1
            
            # update gradient
            g_a_norm = lr * v_a_norm / (np.sqrt(s_a_norm) + epsilon)
            g_b_norm = lr * v_b_norm / (np.sqrt(s_b_norm) + epsilon)
            
            # update params
            a -= g_a_norm
            b -= g_b_norm
            
            a_list.append(a)
            b_list.append(b)
    return a_list, b_list


def adam_matrix(inits, X, Y, lr=0.01, n_iter=10, beta1=0.9, beta2=0.999, epsilon=1e-6):
    n = len(X)
    a, b = inits
    grad_func = [lambda x, y: -2*x*(y-(a*x+b)), lambda x, y: -2*(y-(a*x+b))]
    v = np.array([0, 0])
    s = np.array([0, 0])
    a_list, b_list = [a], [b]
    for _ in range(n_iter):
        t = 1
        for i in range(n):
            x_i, y_i = X[i], Y[i]
            grad = np.array([f(x_i, y_i) for f in grad_func])
            # compute the first moment
            v = beta1 * v + (1-beta1)*grad
            # compute the second moment
            s = beta2*s + (1-beta2)*(grad**2)
            
            # normalisation
            v_norm = v/(1 - np.power(beta1, t))
            s_norm = s/(1 - np.power(beta1, t))
            t += 1
            
            # update gradient
            grad_norm = lr*v_norm/(np.sqrt(s_norm) + epsilon)
            # update params
            a -= grad_norm[0]
            b -= grad_norm[1]
            
            a_list.append(a)
            b_list.append(b)
    return a_list, b_list
