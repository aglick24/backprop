import numpy as np


def logsumexp(Z, axis=1):
    """
    Z - an ndarray
    axis - the dimension over which to logsumexp
    returns:
        logsumexp over the axis'th dimension; returned tensor has same ndim as Z
    """
    maxes = np.max(Z, axis=axis, keepdims=True)
    return maxes + np.log(np.exp(Z - maxes).sum(axis, keepdims=True))


def score(X, theta):
    """
    X - bsz x D_1
    theta - K x D_1
    returns: bsz x K
    """
    return np.matmul(X, theta.transpose())


def xent(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """
    ## TODO: your code here
    scores = score(X, theta)
    return (np.multiply(-Y, scores)).sum(axis=1) + np.log(np.exp(scores).sum(axis=1))


def grad_theta_xent(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of xent(X, Y, theta).sum() wrt theta
    """
    ## TODO: your code here
    K = len(theta)
    D = len(X[0])
    B = len(X)
    scores = score(X, theta)

    matrix = np.exp(scores) / (np.exp(scores).sum(axis=1)).repeat(K, axis=0).reshape(B, K) - Y

    return (np.array(np.hsplit(np.tile(matrix, D), D)).transpose()*np.array(np.hsplit(np.tile(X, K), K))).sum(axis=1)


def mse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """
    ## TODO: your code here
    K = len(theta)
    scores = score(X, theta)

    return ((Y - scores) ** 2).sum(axis=1) / K


def grad_theta_mse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of mse(X, Y, theta).sum() wrt theta
    """
    ## TODO: your code here
    K = len(theta)
    D = len(X[0])
    scores = score(X, theta)

    matrix = -2 / K * (Y - scores)

    return (np.array(np.hsplit(np.tile(matrix, D), D)).transpose()*np.array(np.hsplit(np.tile(X, K), K))).sum(axis=1)


def softmse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """
    ## TODO: your code here
    B = len(X)
    K = len(theta)

    scores = score(X, theta)

    matrix = (np.exp(scores).sum(axis=1)).repeat(K, axis=0).reshape(B, K)

    return ((Y - np.exp(scores) / matrix) ** 2).sum(axis=1) / K


def grad_theta_softmse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of softmse(X, Y, theta).sum() wrt theta
    """
    ## TODO: your code here
    B = len(X)
    K = len(theta)
    D = len(X[0])

    scores = score(X, theta)

    summation = (np.exp(scores).sum(axis=1)).repeat(K, axis=0).reshape(B, K)
    m1 = Y - np.exp(scores)/summation
    m2 = np.exp(scores)/summation**2
    m3 = np.ones_like(summation)/summation

    large_sum = (np.multiply(m1, m2).sum(axis=1)).repeat(K, axis=0).reshape(B, K)

    matrix = large_sum - np.multiply(m1, m3)

    matrix = 2/K * np.multiply(np.exp(scores), matrix)

    return (np.array(np.hsplit(np.tile(matrix, D), D)).transpose()*np.array(np.hsplit(np.tile(X, K), K))).sum(axis=1)



def myloss(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """
    ## TODO: your code here
    K = len(theta)
    scores = score(X, theta)

    return ((Y - np.exp(scores)) ** 2).sum(axis=1) / K



def grad_theta_myloss(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of myloss(X, Y, theta).sum() wrt theta
    """
    ## TODO: your code here
    K = len(theta)
    D = len(X[0])

    scores = score(X, theta)

    matrix = -2/K * (Y - np.exp(scores)) * np.exp(scores)

    return (np.array(np.hsplit(np.tile(matrix, D), D)).transpose()*np.array(np.hsplit(np.tile(X, K), K))).sum(axis=1)