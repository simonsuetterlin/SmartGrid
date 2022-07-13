# from rk4step import rk4step
import numpy as np
import casadi as ca
from scipy import stats
from Code.grid_optimizer import GridOptimizer
from Code.simulation import Simulator
from Code.model import GridModel
from Code.help_functions import *

# set constants: prices, state-space, decision-space
# and max expected change rate of consumption
P_e = 20
P_i = 10
P_b = 5
U = [-2, -1, 0, 1, 2]
O = np.arange(0, 10)
V = np.arange(0, 10)#, step=2)
B = np.arange(6)
V_max_change = 4
B_max_charge = max(B)
num_sub_timepoints = 10
sub_max_change = 2


def L_i(x1):
    return x1[0] * P_i


def L_b(x1):
    return P_b / num_sub_timepoints * sum([battery_usage(x1[0], x1[1][i], x1[2][i]) for i in range(num_sub_timepoints)])


def L_e(x1):
    return P_b / num_sub_timepoints * sum([deficit_O(x1[0], x1[1][i]) - battery_usage(x1[0], x1[1][i], B_max_charge) for i in range(num_sub_timepoints)])


if __name__ == '__main__':
    model = GridModel(L_list=[L_i, L_e, L_b], P_i=P_i, P_e=P_e, P_b=P_b,
                       U=U, O=O, V=V, B=B, V_max_change=V_max_change, sub_max_change=sub_max_change,
                       num_sub_timepoints=num_sub_timepoints, distribution="binom", sub_distribution="uniform")
    grid_opt = GridOptimizer(model)

    grid_opt.calculate_cost_to_go_matrix_sequence(depth=5)

    print(grid_opt.opt_dec_m)
    print(grid_opt.cost_to_go_m)

    # simulate model
    #s = Simulator(model, grid_opt.opt_dec_m)
    #s.simulate(T=100)
    #s.plot_path()
