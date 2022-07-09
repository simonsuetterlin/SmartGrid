#from rk4step import rk4step
import numpy as np
import casadi as ca
from scipy import stats
from grid_optimizer import GridOptimizer
from simulation import Simulator

# set constants: prices, state-space, decision-space
# and max expected change rate of consumption
P_e = 20
P_i = 10
P_b = 5
U = [-2, -1,0,1,2]
O = list(range(13))
V = list(range(13))
B = list(range(4))
V_max_change = 3


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
    def __init__(self, L_list, P_i, P_e, P_b, U, O, V, B, distribution):
        self.L_list = L_list
        self.P_i = P_i
        self.P_e = P_e
        self.P_b = P_b
        self.U = U
        self.O = O
        self.V = V
        self.B = B
        self.dim_O = len(O)
        self.dim_V = len(V)
        self.dim_B = len(B)
        self.dim = (self.dim_O, self.dim_V, self.dim_B)
        self.state = (self.O, self.V, self.B)
        self.set_distribution(distribution)     
    
    def set_distribution(self, distribution):
        """
        Set distribution of city consumption change. Eather uniform or binom
        """
        if distribution == "uniform":
            self.distribution = stats.randint(low=-V_max_change, high=V_max_change+1)
        elif distribution == "binom":
            # binomial distribution around 0 (quicker then own implementation)
            self.distribution = stats.binom(n = 2 * V_max_change, p = 0.5, loc = -V_max_change)
        else: 
            raise ValueError("WRONG DISTRIBUTION NAME")

    def L(self, x0, x1):
        """
        Calculates the loss in the next intervall based on current and
        next state.
        
        Args:
            x0: current state
            x1: next state
        """
        #L_i_out = self.L_i(x0[0], x1[0])
        #L_e_out = self.L_e(x0[0], x1[0], x1[1])
        # assert L_i_out >= 0 or L_e_out >= 0, "LOSS NEGATIVE, {}, {}".format( L_e_out, L_i_out)
        # assert L_i_out + L_e_out >= self.P_i * x1[1], "MIN EXEPTION, {}, {}, {}, {}".format(x0,x1, L_e_out, L_i_out)
        #return L_i_out + L_e_out
        return np.sum([l(x0, x1) for l in self.L_list])
    
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
        
        deficit = deficit_O(x0, (o1, v1, 0))
        overflow_prev, overflow_after = overflow_O(x0, (o1, v1, 0))
        b1 = int(min(max(x0[2] - deficit + overflow_prev, 0) + overflow_after, np.max(self.B)))
        
        return (o1, v1, b1)



def produce_O(x0, x1):
    return 0.5 * (x0[0] + x1[0])

def overflow_O(x0, x1):
    o0 = x0[0]
    o1, v1 = x1[:2]
    if(o0 > v1 and o1 > v1):
        return [0., 0.5*(o0 + o1 - 2 * v1)]
    elif(o0 <= v1 and o1 <= v1):
        return [0., 0.]
    else:
        t = (v1 - o0)/(o1-o0)
        if(o0 <= v1 and o1 > v1):
            return [0., 0.5 * (o1 - v1) * (1 - t)]
        elif(o0 > v1 and o1 <= v1):
            return [0.5 * (o0 - v1) * t, 0.]

def deficit_O(x0, x1):
    return x1[1] + sum(overflow_O(x0, x1)) - produce_O(x0, x1)        

def battery_usage(x0, x1):
    deficit = deficit_O(x0, x1)
    overflow = overflow_O(x0, x1)
    return min(x0[2] + overflow[0], deficit)

def L_e(x0, x1):
    return (deficit_O(x0, x1) - battery_usage(x0, x1)) * P_e

def L_i(x0, x1):
    return produce_O(x0, x1) * P_i

def L_b(x0, x1):
    return battery_usage(x0, x1) * P_b


if __name__ == '__main__':
    model = Model(L_list=[L_i, L_e, L_b], P_i=P_i, P_e=P_e, P_b=P_b, U=U, O=O, V=V, B=B, distribution="binom")
    grid_opt = GridOptimizer(model)
    grid_opt.calculate_cost_to_go_matrix_sequence(depth = 5)

    # print(grid_opt.opt_dec_m)
    # print(grid_opt.cost_to_go_m)

    # simulate model
    s = Simulator(model, grid_opt.opt_dec_m)
    s.simulate(T=100)
    s.plot_path()

"""
def O_i(k, O_i_prev, u):
    return O_i_prev + u * k

def L_i(k, O_i_prev, u):
    return P_i * O_i(k, O_i_prev, u)

def O_e(k, O_i_prev, u, v):
    return np.max([v - O_i(k, O_i_prev, u), 0])

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
