#from rk4step import rk4step
import numpy as np
import casadi as ca
import random


class GridOptimizer:
    """
    Class that optimizes a grid model
    
    Arg:
        model(GridModel): the model that should be controlled
        
    Methods:
        
    """
    def __init__(self, model):
        #self.current_state = None
        self.L = model.L
        self.f = model.f
        self.U = model.U
        self.dim_O = model.dim_O
        self.dim_V = model.dim_V
        self.O = model.O
        self.V = model.V
        self.distribution = model.distribution
        self.iter_depth = 0
        self.cost_to_go_m = None
        self.opt_dec_m = None

    def calculate_cost_to_go_matrix_sequence(self, depth):
        """
        Calculates the cost to go matrix to the specified depth
        
        Args:
            depth
        
        Sets(attributes):
            self.iter_depth(int): new depth
            self.cost_to_go_m: calculated cost to go matrix for this depth
            self.opt_dec_m: calculated optimal decision for this depth
        """
        if self.iter_depth < depth:
            #M = np.zeros((2, self.dim_O,self.dim_V), dtype=float)
            # initialize matricies
            choice = np.zeros((self.dim_O, self.dim_V), dtype=int)
            M = [np.zeros((self.dim_O, self.dim_V), dtype=float), np.zeros((self.dim_O, self.dim_V), dtype=float)]
            # iterately calculates the cost to go matrix from saved depth
            for i in range(depth - self.iter_depth+1):
                if(i != 0):
                    M[i%2], choice = self.calculate_cost_to_go_matrix(M[(i-1)%2])
                else:
                    if self.iter_depth > 0:
                        M[0] = self.cost_to_go_m
                    else:
                        M[0] = self.calculate_cost_to_go_matrix_final_step()
            # set calculated attributes
            self.cost_to_go_m =  M[(depth - self.iter_depth)%2]
            self.iter_depth = depth
            self.opt_dec_m = choice

    # possTODO change end score
    def calculate_cost_to_go_matrix_final_step(self):
        """
        Calculates terminal costs of all states.
        
        Returns:
            matrix with terminal costs
        """
        # initialize matrix
        M = np.zeros((self.dim_O, self.dim_V), dtype=float)
        #for o_index in range(self.dim_O):
        #    for v_index in range(self.dim_V):
        #        M[o_index, v_index] = 0
        #        # end costs are zeros at the moment, possible change
        #        # e.g. M[o_index,v_index] = np.abs(o_index - v_index) * (2 if o_index - v_index > 0 else 1)
        return M

    # TODO optimieren mit list comprehension?
    def calculate_cost_to_go_matrix(self, cost_matrix):
        """
        Evaluates the cost to go function for every possible state
        based on pevious cost to go function.
        
        Args:
            cost_matrix: matrix of previous cost to go function
        
        Returns:
            cost and decision matrix
        """
        # initialize matricies
        M = np.ndarray((self.dim_O, self.dim_V), dtype=float)
        choice = np.ndarray((self.dim_O, self.dim_V), dtype=int)
        # iterates over lines and rows
        for o_index in range(self.dim_O):
            for v_index in range(self.dim_V):
                # calculates current state
                x0 = (self.O[o_index], self.V[v_index])
                # calculates expected cost for this state and optimal decision
                M[o_index,v_index], choice[o_index,v_index] = self.cost_to_go(x0, cost_matrix)
        return M, choice
        # [[cost_to_go(, U, cost_matrix) for o in range(dim_O)] for v in range(dim_V)]

    def cost_to_go(self, x0, cost_matrix):
        """
        Calculate minimal expected loss for a given state.
        
        Args:
            x0: current state
            cost_matrix: cost to go matrix ater the step
        
        Returns:
            min expected loss and corresponding control
        """
        # calculates the path cost for every possible control
        step_cost_to_go_array = [self.calculate_path_cost(x0, u, cost_matrix) for u in self.U]
        # get index of minimum
        min_index = np.argmin(step_cost_to_go_array)
        # returns minimal costs and the control to the minimal cost
        return step_cost_to_go_array[min_index], self.U[min_index]

    def calculate_path_cost(self, x0, u, cost_matrix):
        """
        Calculates the costs after one decision as expected value
        with random variable "next possible state".
        
        Args:
            x0: current state
            u: decision
            cost_matrix: cost to go matrix ater the step
            
        Returns:
            expected loss or np.inf, if state doesn't allow control u
        """
        if x0[0] + u in self.O:
            def rv_loss(v):
                """
                Calculates loss based on next conumption.
                Used as a random variable, since change of consumption
                is a random variable.
                
                Args:
                    v: change of consumption
                Returns:
                    loss
                """
                x1 = self.f(x0, u, v)
                return self.L(x0, x1) + cost_matrix[x1]
            # calculate expected value of rv_loss based on given distribution
            return self.distribution.expect(np.vectorize(rv_loss))
        return np.inf
