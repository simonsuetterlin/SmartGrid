from scipy import stats
import numpy as np
from code.help_functions import battery_usage, overflow_O

class GridModel: 
    """
    The class Model collects all informations about a certain grid-model
    and ist used as input for the class GridOptimizer.
    
    Args:
        L_i, L_e: loss functions
        P_i, P_e, P_b: costs
        U, O, V, B (list): spaces of decision, output and consumption
        distribution: expected distribution of the change of the consumption
        
    Methods:
        L: calculates loss function of the whole model
        f: calculates the next state of the model
    """
    def __init__(self, L_list, P_i, P_e, P_b, U, O, V, B, V_max_change, distribution):
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
        self.B_max_charge = max(B)
        self.V_max_change = V_max_change
        self.dim = (self.dim_O, self.dim_V, self.dim_B)
        self.state = (self.O, self.V, self.B)
        self.distribution_name = distribution
        self.set_distribution()
    
    def __str__(self):
        return (
            f"This grid model has following dimensions and assumptions:\n"
            f"{'Control:':>15}\t({np.min(self.U)}, {np.max(self.U)}, {len(self.U)})\n"
            f"{'Consum:':>15}\t({np.min(self.V)}, {np.max(self.V)}, {self.dim_V})\n"
            f"{'Output:':>15}\t({np.min(self.O)}, {np.max(self.O)}, {self.dim_O})\n"
            f"{'Battery:':>15}\t({np.min(self.B)}, {np.max(self.B)}, {self.dim_B})\n"
            f"{'Distribution:':>15}\t{self.distribution_name}"
                   )
    
    def set_distribution(self):
        """
        Set distribution of city consumption change. Eather uniform or binom
        """
        if self.distribution_name == "uniform":
            self.distribution = stats.randint(low=-self.V_max_change, high=self.V_max_change+1)
        elif self.distribution_name == "binom":
            # binomial distribution around 0 (quicker then own implementation)
            self.distribution = stats.binom(n = 2 * self.V_max_change, p = 0.5, loc = -self.V_max_change)
        else: 
            raise ValueError("This distribution is not yet implemented.")

    def L(self, x0, x1):
        """
        Calculates the loss in the next intervall based on current and
        next state.
        
        Args:
            x0: current state
            x1: next state
        """
        L_eval = np.array([l(x0, x1) for l in self.L_list])
        assert np.all(L_eval >= 0), "LOSS NEGATIVE, {}".format( L_eval)
        return L_eval.sum()
    
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
        
        overflow = sum(overflow_O(x0, (o1, v1, 0)))
        battery_used = battery_usage(x0, (o1, v1, 0), self.B_max_charge)
        # calculates new battery charge:
        # current state + overflow of own power plant - drain of battery
        # but respecting that it can't be overcharged
        # for simplicity b1 is kept as integer. Since there is loss
        # when charging we always round to the smaller integer.
        b1 = int(min(x0[2] + overflow - battery_used, max(self.B)))
        
        return (o1, v1, b1)
