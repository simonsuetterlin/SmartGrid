import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain


# TODO Compare models 
class Simulator:
    """
    Class that simulates the model with the calculated optimal step matrix.
    
    Args:
        model: model that gets simulated
        optimal_step_matrix: calculated optimal step matrix for the model
        f_path: stores be the history of f from simulation
        L_path: stores be the history of L from simulation

    Methods:
        simulate: simulates the model for T steps
        plot: plots the results
    """

    def __init__(self, model, optimal_step_matrix):
        self.model = model
        self.optimal_step_matrix = optimal_step_matrix
        self.f_path = []
        self.L_path = []
        self.battery_path = []

    def simulate(self, T, O_start=None, V_start=None, B_start=0, V_realisation=None):
        """
        Simulates the model for T steps 
        
        Args:
            T: current state
            O_start: initialization of O
            V_start: initialization of V
            B_start: initialization of B
            V_realisation: realisation of the random walk of V
        """
        # reset path if it has been calculated before
        self.f_path = []
        self.L_path = []
        self.battery_path = [B_start]
        # if no realisation of V is given it gets calculated randomly
        if V_realisation is None:
            V_realisation = self.model.distribution.rvs(size=T+1)
            V_sub_realisation = [self.model.sub_distribution.rvs(size=10) for i in range(T+1)]
        # if no initial value is set it get initialized randomly
        if V_start is None:
            V_start = random.randint(min(self.model.V), max(self.model.V))
        if O_start is None:
            O_start = np.min(self.model.O)
        # random initialisation of the model
        x0 = (O_start, V_realisation[0] + [0 for i in range(self.model.num_sub_timepoints)], [B_start for i in range(self.model.num_sub_timepoints)])
        self.f_path.append(x0)
        # simulate the model for every step
        for i in range(T):
            # get optimal step from matrix
            index = self.model.state_to_index([x0[0], x0[1], x0[2]])
            u = self.optimal_step_matrix[tuple(index)]
            v = [V_realisation[i+1]] + V_sub_realisation[i+1]
            x1 = self.model.f([x0[0], x0[1][-1], x0[2][-1]], u, v)
            # save parameters
            self.f_path.append(self.model.f([x0[0], x0[1][-1], x0[2][-1]], u, v))
            self.L_path.append(self.model.L(x1))
            self.battery_path.append(x1[2])
            x0 = x1
        # TODO
        # self.battery_output = [0]
        # print([np.max(0, self.battery_path[i] - self.battery_path[i+1]) for i in range(len(self.battery_path)-1)])
        # self.battery_output.extend([np.max(0, self.battery_path[i] - self.battery_path[i+1]) for i in range(len(self.battery_path)-1)])

    def plot_path(self):
        """
        Plot the simulation results
        """
        min_loss = self.get_min_loss()
        max_loss = self.get_max_loss()

        print(self.model)
        # item for sublist in list_of_lists for item in sublist
        fig, axs = plt.subplots(2, 1, dpi=100, figsize=(20, 9))
        V = list(chain(*[i[1] for i in self.f_path[1:]])) #[item for a in (i[1] for i in self.f_path) for item in a]
        B = list(chain(*self.battery_path[1:])) #[item for b in (i for i in self.battery_path) for item in b]
        t = range(len([i[0] for i in self.f_path]))
        T = range(len(V))
        # Path of O and V
        axs[0].plot(t, [i[0] for i in self.f_path], label="O")
        axs[0].step(T, V, label="V")
        # axs[0].step(t, self.battery_output, label="B")
        axs[0].set_ylabel('O and V path')
        axs[0].legend()
        axs[0].grid(True)
        # battery state
        axs[1].step(range(len(B)), B, label="battery charge")
        axs[1].set_ylabel('storage capacity')
        axs[1].legend()
        axs[1].grid(True)
        plt.show()

    def get_min_loss(self):
        # Der Minimale loss bei maximaler Batterie ladung
        # max_B_use = min(self.f_path[i][1], self.model.B_max_charge)
        # L_b = self.model.P_b * max_B_use
        # L_i = self.model.P_i * (self.f_path[i][1] - max_B_use)
        # f = lambda x: self.model.P_i * x - (self.model.P_i - self.model.P_b) * min(x, self.model.B_max_charge)
        # return [f(self.f_path[i][1]) for i in range(len(self.L_path))]
        # Das ist eigentlich der intern gedeckte Loss.
        return [self.model.P_i * self.f_path[i][1] for i in range(len(self.L_path))]

    def get_max_loss(self):
        # Loss ohne internes Kraftwerk, also ohne Steuerung.
        # ACHTUNG:
        # Realer loss kann über max loss sein, da Steuerung auch mal schlecht.
        return [self.model.P_e * self.f_path[i][1] for i in range(len(self.L_path))]

    def get_subtracted_loss(self, min_loss):
        return [self.L_path[i] - min_loss[i] for i in range(len(self.L_path))]
