#from rk4step import rk4step
import numpy as np
import casadi as ca
from scipy import stats
from grid_optimizer import GridOptimizer
from simulation import Simulator

# set constants: prices, state-space, decision-space
# and max expected change rate of consumption
P_e = 12
P_i = 10
U = [-2, -1,0,1,2]
O = list(range(20))
V = list(range(20))
V_max_change = 2


class Model: 
    """
    The class Model collects all informations about a certain grid-model
    and ist used as input for the class GridOptimizer.
    
    Args:
        L_i, L_e: loss functions
        P_i, P_e: costs
        U, O, V (list): spaces of decision, output and consumption
        distribution: expected distribution of the change of the consumption
        
    Methods:
        L: calculates loss function of the whole model
        f: calculates the next state of the model
    """
    def __init__(self, L_i, L_e, P_i, P_e, U, O, V, distribution):
        self.L_i = L_i
        self.L_e = L_e
        self.P_i = P_i
        self.P_e = P_e
        self.U = U
        self.dim_O = len(O)
        self.dim_V = len(V)
        self.O = O
        self.V = V
        self.distribution = distribution
    
    def L(self, x0, x1):
        """
        Calculates the loss in the next intervall based on current and
        next state.
        
        Args:
            x0: current state
            x1: next state
        """
        L_i_out = self.L_i(x0[0], x1[0])
        L_e_out = self.L_e(x0[0], x1[0], x1[1])
        # assert L_i_out >= 0 or L_e_out >= 0, "LOSS NEGATIVE, {}, {}".format( L_e_out, L_i_out)
        # assert L_i_out + L_e_out >= self.P_i * x1[1], "MIN EXEPTION, {}, {}, {}, {}".format(x0,x1, L_e_out, L_i_out)
        return L_i_out + L_e_out
    
    def f(self, x0, u, v):
        """
        Calculates next state based on current state, decision and new consum.
        
        Args:
            x0: current state
            u: decision (change in output)
            v: change in consum
        """
        o1 = x0[0] + u
        # prohibits stearing out of bounds!
        if o1 not in self.O:
            raise ValueError('Value of the output is out of bounds by current control.')

        # prohibits getting out of bounds, based on 
        v1 = min(max(self.V), max(x0[1] + v, min(self.V)))
        
        return (o1, v1)
    
def discreet_uniform_distibution(a, b):
    xk = np.arange(a, b+1)
    pk = [1/len(xk) for i in range(len(xk))]
    return stats.rv_discrete(name='uniform', values=(xk, pk))

def L_e(o0, o1, v):
    if(o0 >= v and o1 >= v):
        return 0.0
    elif(o0 < v and o1 < v):
        return 0.5*(v - o0 + v - o1) * P_e
    else:
        t = (v - o0)/(o1-o0)
        if(o0 < v and o1 >= v):
            return 0.5 * (v - o0) * t * P_e
        elif(o0 >= v and o1 < v):
            return 0.5 * (v - o1) * (1 - t) * P_e 

def L_i(o0, o1):
    return 0.5 * (o0 + o1) * P_i



uniform = discreet_uniform_distibution(a=-V_max_change, b=V_max_change)
model = Model(L_i=L_i, L_e=L_e, P_i=P_i, P_e=P_e, U=U, O=O, V=V, distribution=uniform)
grid_opt = GridOptimizer(model)
grid_opt.calculate_cost_to_go_matrix_sequence(depth = 5)
#print(grid_opt.opt_dec_m)
#print(grid_opt.cost_to_go_m)

# simulate model
s = Simulator(model, grid_opt.opt_dec_m)
s.simulate(T=100)
s.plot_path()

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
