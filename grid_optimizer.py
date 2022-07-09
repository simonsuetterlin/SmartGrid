#from rk4step import rk4step
import numpy as np
import casadi as ca
import random

class GridOptimizer:
    def __init__(self, model):
        #self.current_state = None
        self.L = model.L
        self.f = model.f
        self.U = model.U
        self.dim_O = model.dim_O
        self.dim_V = model.dim_V
        self.O_range = model.O_range
        self.V_range = model.V_range
        self.distribution = model.distribution
        self.max_cost_to_go_m = np.zeros((self.dim_O, self.dim_V), dtype=float)
        self.max_iter_depth = 0
        self.max_opt_dec_m = None

    def calculate_cost_to_go_matrix_sequence(self, depth):
        if self.max_iter_depth < depth:
            #M = np.zeros((2, self.dim_O,self.dim_V), dtype=float)
            choice = np.zeros((self.dim_O, self.dim_V), dtype=int)
            M = [np.zeros((self.dim_O, self.dim_V), dtype=float), np.zeros((self.dim_O, self.dim_V), dtype=float)]
            M[0] = self.max_cost_to_go_m
            for i in range(depth - self.max_iter_depth+1):
                if(i != 0):
                    M[i%2], choice = self.calculate_cost_to_go_matrix(M[(i-1)%2])
                else:
                    if self.max_iter_depth > 0:
                        continue
                    M[i] = self.calculate_cost_to_go_matrix_final_step()
            self.max_cost_to_go_m =  M[(depth - self.max_iter_depth)%2]
            self.max_iter_depth = depth
            self.max_opt_dec_m = choice

    # calculate matrix with includes indexes for actions U that are optimal for given x0 in matrix
    def calculate_optimal_step_matrix(self, depth):
        if self.max_iter_depth < depth:
            self.calculate_cost_to_go_matrix_sequence(depth=depth)

    # TODO change end score
    def calculate_cost_to_go_matrix_final_step(self):
        M = np.ndarray((self.dim_O, self.dim_V), dtype=float)
        for o_index in range(self.dim_O):
            for v_index in range(self.dim_V):
                M[o_index,v_index] = 0
                # end costs are zeros at the moment, possible change
                # e.g. M[o_index,v_index] = np.abs(o_index - v_index) * (2 if o_index - v_index > 0 else 1)
        return M

    # TODO optimieren mit list comprehension?
    def calculate_cost_to_go_matrix(self, cost_matrix):
        M = np.ndarray((self.dim_O, self.dim_V), dtype=float)
        choice = np.ndarray((self.dim_O, self.dim_V), dtype=int)
        for o_index in range(self.dim_O):
            for v_index in range(self.dim_V):
                x0 = (self.O_range[o_index], self.V_range[v_index])
                M[o_index,v_index], choice[o_index,v_index] = self.cost_to_go(x0, cost_matrix)
        return M, choice
        # [[cost_to_go((o,v), U, cost_matrix) for o in range(dim_O)] for v in range(dim_V)]

    def cost_to_go(self, x0, cost_matrix):
        step_cost_to_go_array = np.zeros(len(self.U), dtype=float)
        for u_index in range(0,len(self.U)):
            step_cost_to_go_array[u_index] = self.calculate_path_cost(x0, self.U[u_index], cost_matrix)
        # step_cost = [self.calculate_path_cost(x0, self.U[u_index], cost_matrix) for u in self.U]
        # min_index = np.argmin(step_cost)
        # return step_cost[min_index], self.U[min_index]
        return np.min(step_cost_to_go_array), self.U[np.argmin(step_cost_to_go_array)]

    #TODO ERWARTUNGSWERT
    def calculate_path_cost(self, x0, u, cost_matrix):
        # Calculate costs of one decision as expected value with random variable "next possible state".
        if 0 <= x0[0] + u < self.dim_O:
            # calculate expected value
            summ = 0
            num_v_in_range = 0
            #rv_f = lambda k: self.L(x0, self.f(x0, u, k)) + cost_matrix[self.f(x0, u, k)]
            def rv_f(k):
                if isinstance(k, np.ndarray):
                    ls = []
                    for i in k:
                        ls.append(rv_f(i))
                    return np.array(ls)
                x1 = self.f(x0, u, k)
                return self.L(x0, x1) + cost_matrix[x1]
            return self.distribution.expect(rv_f)
            # currently uniform distribution over all V capped at [0, dim_V]
            # so it is no uniform distribution!!
        return np.inf
