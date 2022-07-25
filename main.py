# from rk4step import rk4step
import numpy as np
import pickle
from sqlalchemy import false
from yaml import load
from src.grid_optimizer import GridOptimizer
from src.simulation import Simulator
from src.model import GridModel
from src.help_functions import *
from src.markov_chain import init_chain

LOAD_OPTIMIZER = False
load_name = "model_smaller_output"

SAVE_OPTIMIZER = False
save_name = "model_smaller_output"

USE_REAL_DATA = True

# set constants: prices, state-space, decision-space
# and max expected change rate of consumption
P_e = 30
P_i = 10
P_b = 5
U = np.array([- np.inf,-2, -1, 0, 1, 2, np.inf]) # -inf stands for instant shutdown inf for instant start to 80%
O = np.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) #np.arange(0, 12)
V = np.arange(21)
B = np.arange(5)
V_max_change = 4
B_max_charge = max(B)

# instant change of O
def L_i_instant(x0, x1):
    split_O_into_two = [O[i:i+int(len(O)/2)+1] for i in range(0, len(O), int(len(O)/2)+1)]
    prod_O=produce_O_instant(x0, x1)
    if x0[0] == 0:
        return prod_O * P_i * 1.5
    elif x0[0] in split_O_into_two[0]:
        return prod_O * P_i * 1.25
    elif x0[0] in split_O_into_two[1]:
        return prod_O * P_i
    else:
        raise ValueError("x0[0] not in O split array")

def L_b_instant(x0, x1):
    return battery_usage_instant(x0, x1) * P_b

def L_e_instant(x0, x1):
    return (deficit_O_instant(x0,x1) - battery_usage_instant(x0, x1)) * P_e


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

def save_optimizer(optimizer, name):
    with open(f'./optimizer_weights/{name}.pkl', 'wb') as outp:
        pickle.dump(optimizer, outp, pickle.HIGHEST_PROTOCOL)

def load_optimizer(name):
    with open(f'./optimizer_weights/{name}.pkl', 'rb') as inp:
        grid_opt = pickle.load(inp)
    return grid_opt


if __name__ == '__main__':
    if not LOAD_OPTIMIZER:
        chain = init_chain(np.max(V))
        
        model = GridModel(L_list=[L_i_instant, L_e_instant, L_b_instant], P_i=P_i, P_e=P_e, P_b=P_b,
                        U=U, O=O, V=V, B=B, V_max_change=V_max_change,
                        chain=chain)
        grid_opt = GridOptimizer(model)
        grid_opt.calculate_cost_to_go_matrix_sequence(depth=5)
        if SAVE_OPTIMIZER:
            save_optimizer(grid_opt, save_name)
    else:
        print(f"Load model file {load_name}...", end='\r')
        grid_opt = load_optimizer(load_name)
        print(f"Model {load_name} loaded and ready to go.\t\t\t\t")

    # simulate model
    s = Simulator(grid_opt.model, grid_opt.opt_dec_m)
    s.simulate(T=150, real_data=USE_REAL_DATA)
    s.plot_path()

    def simulate(s):
        new_simulation = input("Input an integer n for new simulation of length n: ")
        if new_simulation.isdigit():
            s.simulate(T=int(new_simulation), real_data=USE_REAL_DATA)
            s.plot_path()
            simulate(s)

    simulate(s)

