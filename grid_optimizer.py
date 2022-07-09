#from rk4step import rk4step
import numpy as np
import casadi as ca
import random


class GridOptimizer:
    """
    Class that optimizes a grid model
    
    Arg:
        model(GridModel): the model that should be controlled
        
    Attributes:
        iter_depth: Depth to which the cost-to-go-function has been calculated
        cost_to_go_m: Evaluation of the cost-to-go-function
        opt_dec_m: Optimal decision based on cost_to_go_m
        
    Methods:
        L: loss function
        f: step function
        calculate_cost_to_go_matrix_sequence(depth): 
            evaluates the cost-to-go-function to certain depth
            and saves it to cost_to_go_m        
    """
    def __init__(self, model):
        self.model = model
        self.iter_depth = 0
        self.cost_to_go_m = None
        self.opt_dec_m = None

    def calculate_cost_to_go_matrix_sequence(self, depth):
        """
        Calculates the cost to go matrix to the specified depth.
        
        Args:
            depth
        
        Sets(attributes):
            self.iter_depth(int): new depth
            self.cost_to_go_m: calculated cost to go matrix for this depth
            self.opt_dec_m: calculated optimal decision for this depth
        """
        if self.iter_depth < depth:
            # initialize matricies
            choice = np.zeros(self.model.dim, dtype=int)
            M = [np.zeros(self.model.dim, dtype=float), np.zeros(self.model.dim, dtype=float)]
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
        M = np.zeros(self.model.dim, dtype=float)
        #M = np.array([[func for v_index in range(self.model.dim_V)] for o_index in range(self.model.dim_O)])
        return M

    def calculate_cost_to_go_matrix(self, cost_matrix):
        """
        Evaluates the cost to go function for every possible state
        based on pevious cost to go function.
        
        Args:
            cost_matrix: matrix of previous cost to go function
        
        Returns:
            cost and decision matrix
        """
        # list comprehension to iterate over rows and columns
        matr = np.array([[[self.cost_to_go((self.model.V[v_index], self.model.O[o_index], self.model.B[b_index]), cost_matrix) for b_index in range(self.model.dim_B)] for v_index in range(self.model.dim_V)] for o_index in range(self.model.dim_O)])
        # matr = np.array([self.cost_to_go(tuple([self.model.state[i][j] for i, j in enumerate(index)]), cost_matrix) for index in look_up_table])
        # extract cost-to-go-matrix and decision-matrix
        cost_to_go_matrix = matr[:,:,:,0]
        decision_matrix = matr[:,:,:,1].astype(int)
        return cost_to_go_matrix, decision_matrix
        

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
        step_cost_to_go_array = [self.calculate_path_cost(x0, u, cost_matrix) for u in self.model.U]
        # get index of minimum
        min_index = np.argmin(step_cost_to_go_array)
        # returns minimal costs and the control to the minimal cost
        return step_cost_to_go_array[min_index], self.model.U[min_index]

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
        if x0[0] + u in self.model.O:
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
                x1 = self.model.f(x0, u, v)
                return self.model.L(x0, x1) + cost_matrix[x1]
            # calculate expected value of rv_loss based on given distribution
            return self.model.distribution.expect(np.vectorize(rv_loss))
        return np.inf

    
def look_up_table(dim):
    n = len(dim)
    ls = []
    for i in range(n):
        transposition = list(range(n))
        transposition[i], transposition[n-1] = n-1, i
        x = np.transpose(np.transpose(np.zeros(dim, dtype=int), axes=transposition) + np.arange(dim[i]), axes=transposition)
        ls.append(x)
    return np.stack(ls, axis=n)
