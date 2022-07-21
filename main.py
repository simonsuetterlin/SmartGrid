# from rk4step import rk4step
import numpy as np
from src.grid_optimizer import GridOptimizer
from src.simulation import Simulator
from src.model import GridModel
from src.help_functions import *
from src.markov_chain import chain

# set constants: prices, state-space, decision-space
# and max expected change rate of consumption
P_e = 20
P_i = 10
P_b = 5
U = [- np.inf,-2, -1, 0, 1, 2, np.inf] # -inf stands for instant shutdown inf for instant start to 80%
O = np.arange(0, 12)
V = np.arange(0, 12)
B = np.arange(10)
V_max_change = 4
B_max_charge = max(B)

#factors are efficiency loss at lower output levels
def L_i(x0, x1):
    # seperate O into three sets of the same size to have a weighted score
    split_O_into_three = [O[i:i+int(len(O)/3)+1] for i in range(0, len(O), int(len(O)/3)+1)]
    prod_O=produce_O(x0,x1)
    if x0[0] == 0:
        return prod_O * P_i * 1.75
    elif x0[0] in split_O_into_three[0]:
        return prod_O * P_i * 1.5
    elif x0[0] in split_O_into_three[1]:
        return prod_O * P_i * 1.25
    elif x0[0] in split_O_into_three[2]:
        return prod_O * P_i
    else:
        raise ValueError("x0[0] not in O split array")

def L_b(x0, x1):
    return battery_usage(x0, x1, B_max_charge) * P_b


def L_e(x0, x1):
    return (deficit_O(x0, x1) - battery_usage(x0, x1, B_max_charge)) * P_e


if __name__ == '__main__':
    model1 = GridModel(L_list=[L_i, L_e, L_b], P_i=P_i, P_e=P_e, P_b=P_b,
                       U=U, O=O, V=V, B=B, V_max_change=V_max_change,
                       chain=chain)
    grid_opt1 = GridOptimizer(model1)
    grid_opt1.calculate_cost_to_go_matrix_sequence(depth=5)

    # simulate model
    s = Simulator(model1, grid_opt1.opt_dec_m)
    s.simulate(T=100)
    s.plot_path()

    def simulate(s):
        new_simulation = input("Press 'y' for new simulation: ")
        if new_simulation == 'y' or new_simulation == 'Y':
            s.simulate(T=100)
            s.plot_path()
            simulate(s)

    simulate(s)

