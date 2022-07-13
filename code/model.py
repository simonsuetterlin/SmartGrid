from scipy import stats
import numpy as np
from Code.help_functions import battery_usage, overflow_O
from Code.look_up_table import convert_index_to_state, convert_state_to_index


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

    def __init__(self, L_list, P_i, P_e, P_b, U, O, V, B, V_max_change, sub_max_change, num_sub_timepoints, distribution, sub_distribution):
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
        self.sub_max_change = sub_max_change
        self.dim = (self.dim_O, self.dim_V, self.dim_B)
        self.state_space = (self.O, self.V, self.B)
        self.num_sub_timepoints = num_sub_timepoints
        self.distribution_name = distribution
        self.sub_distribution_name = sub_distribution
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
            self.distribution = stats.randint(low=-self.V_max_change, high=self.V_max_change + 1)
        elif self.distribution_name == "binom":
            # binomial distribution around 0 (quicker then own implementation)
            self.distribution = stats.binom(n=2 * self.V_max_change, p=0.5, loc=-self.V_max_change)
        else:
            raise ValueError("This distribution is not yet implemented.")

        if self.sub_distribution_name == "uniform":
            self.sub_distribution = stats.randint(low=-self.sub_max_change, high=self.sub_max_change + 1)
        elif self.sub_distribution_name == "binom":
            # binomial distribution around 0 (quicker then own implementation)
            self.sub_distribution = stats.binom(n=2 * self.sub_max_change, p=0.5, loc=-self.sub_max_change)
        else:
            raise ValueError("This distribution is not yet implemented.")

    def L(self, x1):
        """
        Calculates the loss in the next intervall based on current and
        next state.
        
        Args:
            x0: current state
            x1: next state
        """
        L_eval = np.array([l(x1) for l in self.L_list])
        assert np.all(L_eval >= 0), "LOSS NEGATIVE, {}".format(L_eval)
        return L_eval.sum()

    def f(self, x0, u, v):
        """
        Calculates next state based on current state, decision and new consum.
        
        Args:
            x0: current state
            u: decision (change in output)
            v: change in consume
        """
        o1 = x0[0] + u
        # prohibits stearing out of bounds!
        if o1 not in self.O:
            raise ValueError('Value of the output is out of bounds by current control.')

        # prohibits getting out of bounds, based on

        v1 = [min(max(self.V), max(x0[1] + v[i], min(self.V))) for i in range(self.num_sub_timepoints)]

        overflow = [overflow_O(o1, v1[i]) for i in range(self.num_sub_timepoints)]
        battery_used = []
        b1 = []
        for i in range(self.num_sub_timepoints):
            if i == 0:
                battery_used.append(battery_usage(o1, v[i], x0[2]))
            else:
                battery_used.append(battery_usage(o1, v[i], b1[i-1]))
            b1.append(int(min(x0[2] + overflow[i] - battery_used[i], max(self.B))))
        # calculates new battery charge:
        # current state + overflow of own power plant - drain of battery
        # but respecting that it can't be overcharged
        # for simplicity b1 is kept as integer. Since there is loss
        # when charging we always round to the smaller integer.

        return (o1, v1, b1)

    def index_to_state(self, index):
        return convert_index_to_state(index, self.state_space)

    def state_to_index(self, state):
        return convert_state_to_index(state, self.state_space)
