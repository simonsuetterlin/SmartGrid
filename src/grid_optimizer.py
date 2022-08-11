# from rk4step import rk4step
import numpy as np
from src.look_up_table import look_up_table


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
        assert depth > 0 and isinstance(depth, int), "Depth must be non negative integer."
        if self.iter_depth < depth:
            print(
                f"Calculating the cost to go matrix and the optimal decision "
                f"matrix to the total depth of {depth} from the depth "
                f"{self.iter_depth}."
            )
            # initialize matricies
            choice = np.zeros(self.model.dim, dtype=int)
            M = [np.zeros(self.model.dim, dtype=float), np.zeros(self.model.dim, dtype=float)]
            # iterately calculates the cost to go matrix from saved depth
            depth_to_go = depth - self.iter_depth + 1
            for i in range(depth_to_go):
                print(f"{f'Reached depth {i} from {depth_to_go - 1}:':<25}\t {100. * i / depth_to_go:5.1f}%", end="\r")
                if i != 0:
                    M[i % 2], choice = self.calculate_cost_to_go_matrix(M[(i - 1) % 2])
                else:
                    if self.cost_to_go_m is not None:
                        M[0] = self.cost_to_go_m
                    else:
                        M[0] = self.calculate_cost_to_go_matrix_final_step()
            # set calculated attributes
            print(f"{'Finished:':<25}\t {100. * depth_to_go / depth_to_go:.1f}%")
            self.cost_to_go_m = M[(depth - self.iter_depth) % 2]
            self.iter_depth = depth
            self.opt_dec_m = choice

    # possTODO change end score
    def calculate_cost_to_go_matrix_final_step(self):
        """
        Calculates terminal costs of all states.
        
        Returns:
            matrix with terminal costs
        """
        # adds cost to charge the batery to max charge
        f = lambda index: (self.model.B_max_charge - self.model.B[index[2]]) * self.model.P_i
        M = np.vectorize(f)(look_up_table(self.model.dim))
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
        f = lambda index: self.cost_to_go(index, cost_matrix)
        matr = np.vectorize(f)(look_up_table(self.model.dim))
        # extract cost-to-go-matrix and decision-matrix
        cost_to_go_matrix = matr[0]
        decision_matrix = matr[1].astype(int)
        return cost_to_go_matrix, decision_matrix

    def cost_to_go(self, index, cost_matrix):
        """
        Calculate minimal expected loss for a given state.
        
        Args:
            index: multi-index representing current state
            cost_matrix: cost to go matrix ater the step
        
        Returns:
            min expected loss and corresponding control
        """
        # calculates the path cost for every possible control
        step_cost_to_go_array = [self.calculate_path_cost(index, u, cost_matrix) for u in self.model.U]
        # get index of minimum
        min_index = np.argmin(step_cost_to_go_array)
        # returns minimal costs and the control to the minimal cost. Get new u function so that infinity will not be in optimal cost matrix
        return step_cost_to_go_array[min_index], self.model.get_new_u(self.model.O[index[0]], self.model.U[min_index])

    def calculate_path_cost(self, index, u, cost_matrix):
        """
        Calculates the costs after one decision as expected value
        with random variable "next possible state".
        
        Args:
            index: multi-index representing current state
            u: decision
            cost_matrix: cost to go matrix ater the step
            
        Returns:
            expected loss or np.inf, if state doesn't allow control u
        """
        state0 = self.model.index_to_state(index)
        # instart start to 80 % and instant shutdown to 0
        u = self.model.get_new_u(state0[0], u)

        if state0[0] + u in self.model.O:
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
                state1 = self.model.f(state0, u, v)
                index1 = self.model.state_to_index(state1)
                return self.model.L(state0, state1) + cost_matrix[index1]

            # calculate expected value of rv_loss based on given distribution
            return self.model.distribution[index[1]].expect(np.vectorize(rv_loss))
        return np.inf
