from rk4step import rk4step
import numpy as np
import casadi as ca

Ts = .5
N = 19
x0 = np.array([1, 0])
t_grid = np.linspace(0, Ts*N, N+1)
t_span = [0, Ts*N]
t0 = 0

def f(t0, x):
    return ca.vertcat(x[1], -.2*x[1] - x[0])

X_rk4 = np.zeros((x0.size, N+1))
X_rk4[:,0] = x0
for k in range(1, len(t_grid)):
    X_rk4[:,k] = rk4step(f, Ts,  X_rk4[:,k-1], t0).full().flatten()