#from rk4step import rk4step
import numpy as np
import casadi as ca
from scipy import stats
from code.grid_optimizer import GridOptimizer
from code.simulation import Simulator
from code.model import GridModel
from code.help_functions import *

# set constants: prices, state-space, decision-space
# and max expected change rate of consumption
P_e = 20
P_i = 10
P_b = 5
U = [-2, -1,0,1,2]
O = np.arange(10, 21)
V = np.arange(10, 21)
B = np.arange(6)
V_max_change = 4
B_max_charge = max(B)

def L_i(x0, x1):
    return produce_O(x0, x1) * P_i

def L_b(x0, x1):
    return battery_usage(x0, x1, B_max_charge) * P_b

def L_e(x0, x1):
    return (deficit_O(x0, x1) - battery_usage(x0, x1, B_max_charge)) * P_e

if __name__ == '__main__':
    model1 = GridModel(L_list=[L_i, L_e, L_b], P_i=P_i, P_e=P_e, P_b=P_b,
                       U=U, O=O, V=V, B=B, V_max_change=V_max_change,
                       distribution="binom")
    #model2 = GridModel(L_list=[L_i, L_e, L_b], P_i=P_i, P_e=P_e, P_b=P_b,
    #                   U=U, O=O, V=V, B=[0], V_max_change=V_max_change,
    #                   distribution="binom")
    grid_opt1 = GridOptimizer(model1)
    #grid_opt2 = GridOptimizer(model2)
    grid_opt1.calculate_cost_to_go_matrix_sequence(depth = 5)
    #grid_opt2.calculate_cost_to_go_matrix_sequence(depth = 5)

    # print(grid_opt.opt_dec_m)
    # print(grid_opt.cost_to_go_m)

    # simulate model
    s1 = Simulator(model1, grid_opt1.opt_dec_m)
    s1.simulate(T=100)
    s1.plot_path()
    #s2 = Simulator(model2, grid_opt2.opt_dec_m)
    #s2.simulate(T=100)
    #s2.plot_path()
