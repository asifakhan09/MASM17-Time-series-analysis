import numpy as np
import pandas as pd

# ----------------------------
# Helpers
# ----------------------------
LAGS = (1, 23, 24)

def _get_columns(data):
    data = np.asarray(data, dtype=float)
    y = data[:, 1]
    air = data[:, 2]
    sup = data[:, 3]
    return y, air, sup

def _build_phi_A(y, t):
    return np.array([1.0, y[t-1], y[t-23], y[t-24]])

def _build_phi_B(y, air, sup, t):
    return np.array([1.0, y[t-1], y[t-23], y[t-24], air[t], sup[t]])

def _fit_ls(Phi, Y):
    theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    return theta

def _indices(payload):
    k = int(payload["k_steps"])
    s = int(payload["start_idx"]) - 1
    e = int(payload["end_idx"]) - 1
    return k, np.arange(s, e + 1)

# ----------------------------
# Part A: AR
# ----------------------------
def solutionA(payload):
    data = np.asarray(payload["data"])
    y, air, sup = _get_columns(data)
    k, idx = _indices(payload)

    maxlag = 24
    fit_end = idx[0]

    Phi, Y = [], []
    for t in range(maxlag, fit_end):
        Phi.append(_build_phi_A(y, t))
        Y.append(y[t])

    theta = _fit_ls(np.vstack(Phi), np.array(Y))

    yhat = []
    for t in idx:
        y_local = y.copy()
        for j in range(1, k + 1):
            tt = t + j
            phi = np.array([1.0, y_local[tt-1], y_local[tt-23], y_local[tt-24]])
            y_local[tt] = phi @ theta
        yhat.append(y_local[t + k])

    return yhat

# ----------------------------
# Part B: ARX
# ----------------------------
def solutionB(payload):
    data = np.asarray(payload["data"])
    y, air, sup = _get_columns(data)
    k, idx = _indices(payload)

    maxlag = 24
    fit_end = idx[0]

    Phi, Y = [], []
    for t in range(maxlag, fit_end):
        Phi.append(_build_phi_B(y, air, sup, t))
        Y.append(y[t])

    theta = _fit_ls(np.vstack(Phi), np.array(Y))

    yhat = []
    for t in idx:
        y_local = y.copy()
        for j in range(1, k + 1):
            tt = t + j
            phi = np.array([1.0, y_local[tt-1], y_local[tt-23], y_local[tt-24],
                            air[t], sup[t]])
            y_local[tt] = phi @ theta
        yhat.append(y_local[t + k])

    return yhat

# ----------------------------
# Part C: Recursive (Kalman / RLS)
# ----------------------------
def solutionC(payload):
    data = np.asarray(payload["data"])
    y, air, sup = _get_columns(data)
    k, idx = _indices(payload)

    maxlag = 24
    q = 1e-4
    r = np.var(y) * 0.01

    theta = np.zeros(6)
    P = np.eye(6) * 100
    Q = np.eye(6) * q

    theta_store = {}

    for t in range(maxlag, idx[-1] + 1):
        phi = np.array([1.0, y[t-1], y[t-23], y[t-24], air[t], sup[t]])
        Pp = P + Q
        K = Pp @ phi / (phi @ Pp @ phi + r)
        e = y[t] - phi @ theta
        theta = theta + K * e
        P = Pp - np.outer(K, phi) @ Pp
        theta_store[t] = theta.copy()

    yhat = []
    for t in idx:
        th = theta_store[t]
        y_local = y.copy()
        for j in range(1, k + 1):
            tt = t + j
            phi = np.array([1.0, y_local[tt-1], y_local[tt-23], y_local[tt-24],
                            air[t], sup[t]])
            y_local[tt] = phi @ th
        yhat.append(y_local[t + k])

    return yhat