import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from src.markov_chain import get_data_numeric


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
        self.number_houses = 0
        self.start_time = ''
        self.f_path = []
        self.L_path = []
        self.battery_path = []

    def simulate(self, T, real_data=False, O_start=0, V_start=None, B_start=0, V_realisation=None):
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
        # if no initial value is set it get initialized randomly
        if V_start is None:
            # V_start = random.randint(min(self.model.V), max(self.model.V))
            V_start_probs = self.model.chain.n_order_matrix(self.model.chain.observed_p_matrix, order=20)[1, :]
            values = (self.model.V, V_start_probs)
            distr = stats.rv_discrete(values=values)
            V_start = distr.rvs(size=1)
            # set a arbitrary start time since random process is not time dependent
            self.start_time = '00:00'

        # if no realisation of V is given it gets calculated randomly
        if V_realisation is None:
            if not real_data:
                V_realisation = self.model.chain.simulate(n=T, start=int(V_start[0]), ret="states")
            else:
                data_numeric, timepoints_data, self.number_houses = get_data_numeric(max(self.model.V))
                start = np.random.randint(0, 200) * 12
                self.start_time = timepoints_data[start][11:16]
                V_realisation = np.array(data_numeric[start:start + T])
                V_start = V_realisation[0]

        if O_start is None:
            O_start = np.min(self.model.O)

        # random initialisation of the model
        x0 = (O_start, V_start, B_start)
        self.f_path.append(x0)
        # simulate the model for every step
        for i in range(T):
            # get optimal step from matrix
            index = self.model.state_to_index(x0)
            u = self.optimal_step_matrix[index]
            v = V_realisation[i]
            x1 = self.model.f(x0, u, v)
            # save parameters
            self.f_path.append(self.model.f(x0, u, v))
            self.L_path.append(self.model.L(x0, x1))
            self.battery_path.append(x1[2])
            x0 = x1

    def plot_path(self):
        """
        Plot the simulation results
        """
        min_loss = self.get_min_loss()
        max_loss = self.get_max_loss()

        print(self.model)

        fig, axs = plt.subplots(3, 1, dpi=100, figsize=(20, 12))
        fig.suptitle(f"simulation for {self.number_houses} houses", fontsize=20)
        t_names = self.get_time_list(len(self.L_path) + 1)
        t = np.arange(len(self.L_path) + 1)
        # Path of O and V
        axs[0].step(t, [i[0] for i in self.f_path], label="output powerplant")
        axs[0].step(t, [i[1] for i in self.f_path], label="district energy consumption")
        # axs[0].step(t, self.battery_output, label="B")
        axs[0].set_xticks(t[0::12], t_names[0::12])
        axs[0].set_ylabel('MW', rotation=0, labelpad=20)
        axs[0].legend(loc='lower right')
        axs[0].grid(True)

        # battery state
        axs[1].plot(t, self.battery_path, label="battery charge")
        axs[1].set_xticks(t[0::12], t_names[0::12])
        axs[1].set_ylabel('MWh', rotation=0, labelpad=30)
        axs[1].legend(loc='lower right')
        axs[1].grid(True)

        # loss
        axs[2].hlines(0, xmin=0, xmax=len(t) - 1, label="min loss", color='black')
        axs[2].step(t[1:], np.cumsum([(self.L_path[i] - min_loss[i]) / 1000 for i in range(len(self.L_path))]),
                    label="cum subtracted loss")
        axs[2].step(t[1:], np.cumsum([(max_loss[i] - min_loss[i]) / 1000 for i in range(len(self.L_path))]),
                    label="cum subtracted max loss")
        axs[2].set_xticks(t[0::12], t_names[0::12])
        axs[2].set_ylabel('k €', rotation=0, labelpad=20)
        axs[2].set_xlabel('time')
        axs[2].legend(loc='upper left')
        axs[2].grid(True)
        plt.show()

    def get_min_loss(self):
        """
        Calculates the minimal possible loss: The power plant O always produces exactly what the consumer needs.
        """
        return [self.model.P_i * 0.5 * self.f_path[i][1] for i in range(len(self.L_path))]

    def get_max_loss(self):
        """
        Calculates the maximal possible loss: The the needs of the consumer get fully covered by the external electricity source.
        """
        return [self.model.P_e * 0.5 * self.f_path[i][1] for i in range(len(self.L_path))]

    def get_subtracted_loss(self, min_loss):
        """
        Subtracts the minimal loss from the general loss
        """
        return [self.L_path[i] - min_loss[i] for i in range(len(self.L_path))]

    def get_time_list(self, length):
        """
        returns a list of the timestamps for the plot
        """
        x = 0
        if self.start_time[3] == '3':
            x = 1
        return [f"{(int(self.start_time[0:2]) + i // 2) % 24 :02}:{3 * (i % 2)}0" for i in range(x, length + x)]
