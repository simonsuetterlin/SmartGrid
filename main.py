#from rk4step import rk4step
import numpy as np
import casadi as ca
from scipy.stats import rv_continuous, rv_discrete
from grid_optimizer import GridOptimizer

P_e = 12
P_i = 10

U = [-2, -1,0,1,2]
O = list(range(20))
V = list(range(20))
V_max_change = 2

class Model:
    def __init__(self, L_i, L_e, U, O, V, distribution):
        self.L_i = L_i
        self.L_e = L_e
        self.U = U
        self.dim_O = len(O)
        self.dim_V = len(V)
        self.O = O
        self.V = V
        self.distribution = distribution
    
    def L(self, x0, x1):
        return self.L_i(x0[0], x1[0]) + self.L_e(x0[0], x1[0], x1[1])
    
    """
    Calculates next state based on decision and new consum.
    """
    def f(self, x0, u, v):
        o1 = x0[0] + u
        v1 = min(len(V)-1, max(x0[1] + v, 0))
        if 0 > o1 or o1 >= len(V):
            return None
        return (o1, v1)
 
    
class UniformDiscretDistr(rv_discrete):
    def _pmf(self, k):
        return 1./(self.b + 1 - self.a)


def L_e(o0, o1, v):
    if(o0 >= v and o1 >= v):
        return 0
    elif(o0 < v and o1 < v):
        return 0.5*(v - o0 + v - o1) * P_e
    else:
        t = (v - o0)/(o1-o0)
        if(o0 < v and o1 >= v):
            return 0.5 * (o1 - v) * (1-t) * P_e
        elif(o0 >= v and o1 < v):
            return 0.5 * (o0 - v) * t * P_e 

def L_i(o0, o1):
    return 0.5 * (o0 + o1) * P_i

uniform = UniformDiscretDistr(a=-V_max_change, b=V_max_change, name="uniform")
model = Model(L_i=L_i, L_e=L_e, U=U, O=O, V=V, distribution=uniform)
grid_opt = GridOptimizer(model)
grid_opt.calculate_optimal_step_matrix(depth = 5)
print(grid_opt.max_opt_dec_m)
#print(grid_opt.max_cost_to_go_m)

"""
def V(v):
    return v * C

def O_i(k, O_i_prev, u):
    return O_i_prev + u * k

def L_i(k, O_i_prev, u):
    return P_i * O_i(k, O_i_prev, u)

def O_e(k, O_i_prev, u, v):
    return np.max([V(v) - O_i(k, O_i_prev, u), 0])

def L_e(k, O_i_prev, u, v):
    return P_e * O_e(k, O_i_prev, u, v)

# integrator nicht mehr wrong, aber wahrscheinlich ineffizient
def L(x0, u, v):


    x = ca.SX.sym('x')
    k = ca.SX.sym('k')
    ode = {'x':x, 't':k,'ode':P_e * ca.fmax(v * C - x0[0] + u * k, 0) + P_i * x0[0] + u * k}

    F = ca.integrator('F', 'idas', ode,{'t0':0,'tf':2})
    r = F(x0=0)
    return(r['xf'])
"""
