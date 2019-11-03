import numpy as np


def move(p, U):
    """
    movement inaccurate
    p: probability distribution
    U: number of steps moving
    """
    p = np.array(p)
    move_prob = np.array([pOvershoot, pExact, pUndershoot])
    n = len(p)
    U = U % n

    q = []
    for i in range(n):
        steps = [i - U - 1, i - U, i - U + 1]
        q_prob = np.dot(p[steps], move_prob)
        q.append(q_prob)
    return q


def sense(p, Z):
    prob = np.array(p)
    measure = np.array([pHit if i == Z else pMiss for i in world])

    combine_prob = prob * measure

    norm_prob = combine_prob / sum(combine_prob)
    return norm_prob


if __name__ == "__main__":
    p = [0.2, 0.2, 0.2, 0.2, 0.2]
    world = ['green', 'red', 'red', 'green', 'green']
    measurements = ['red', 'green']
    motions = [1, 1]
    pHit = 0.6
    pMiss = 0.2
    pExact = 0.8
    pOvershoot = 0.1
    pUndershoot = 0.1

    for i in range(len(measurements)):
        # sense -> move
        p = sense(p, measurements[i])
        p = move(p, motions[i])
    print(p)
    