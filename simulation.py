import random
import numpy as np
import matplotlib.pyplot as plt


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

    def simulate(self, T, O_start=0, V_start=None, B_start=0, V_realisation=None):
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
        if V_realisation == None:
            V_realisation = self.model.distribution.rvs(size=T)
        # if no initial value is set it get initialized randomly
        if V_start == None:
            V_start = random.randint(min(self.model.V), max(self.model.V))
        # random initialisation of the model    
        x0 = (O_start, V_start, B_start)
        self.f_path.append(x0)
        # simulate the model for every step
        for i in range(T):
            # get optimal step from matrix
            u = self.optimal_step_matrix[x0]
            v = V_realisation[i]
            x1 = self.model.f(x0, u, v)
            # save parameters
            self.f_path.append(self.model.f(x0, u, v))
            self.L_path.append(self.model.L(x0, x1))
            self.battery_path.append(x1[2])
            x0 = x1
        # TODO
        #self.battery_output = [0]
        #print([np.max(0, self.battery_path[i] - self.battery_path[i+1]) for i in range(len(self.battery_path)-1)])
        #self.battery_output.extend([np.max(0, self.battery_path[i] - self.battery_path[i+1]) for i in range(len(self.battery_path)-1)])
            
    def plot_path(self):
        """
        Plot the simulation results
        """
        min_loss = self.get_min_loss()
        max_loss = self.get_max_loss()
        
        print(self.model)

        fig, axs = plt.subplots(5, 1, dpi=100, figsize=(20,9))
        t = np.arange(len(self.L_path) + 1)
        # Path of O and V
        axs[0].plot(t, [i[0] for i in self.f_path], label="O")
        axs[0].step(t, [i[1] for i in self.f_path], label="V")
        #axs[0].step(t, self.battery_output, label="B")
        axs[0].set_ylabel('O and V path')
        axs[0].legend()
        axs[0].grid(True)
        # battery state
        axs[1].step(t,self.battery_path, label="battery charge")
        axs[1].set_ylabel('storage capacity')
        axs[1].legend()
        axs[1].grid(True)
        # Loss paths and possible max/ min loss
        axs[2].step(t[1:], self.L_path, label="loss")
        axs[2].step(t[1:], min_loss, label="min loss")
        axs[2].step(t[1:], max_loss, label="max loss")
        axs[2].set_ylabel('Loss')
        axs[2].legend()
        axs[2].grid(True)
        # loss subtracted by minimal loss
        axs[3].hlines(0, xmin=0, xmax=len(t)-1, label="min loss", color='black')
        axs[3].step(t[1:], [self.L_path[i] - min_loss[i] for i in range(len(self.L_path))], label="subtracted loss")
        axs[3].step(t[1:], [max_loss[i] - min_loss[i] for i in range(len(self.L_path))], label="subtracted max loss")
        axs[3].set_ylabel('Loss')
        axs[3].set_xlabel('time')
        axs[3].legend()
        axs[3].grid(True)
        # cumulated losss substr.
        axs[4].hlines(0, xmin=0, xmax=len(t)-1, label="min loss", color='black')
        axs[4].step(t[1:], np.cumsum([self.L_path[i] - min_loss[i] for i in range(len(self.L_path))]), label="cum subtracted loss")
        axs[4].step(t[1:], np.cumsum([max_loss[i] - min_loss[i] for i in range(len(self.L_path))]), label="cum subtracted max loss")
        axs[4].set_ylabel('Loss')
        axs[4].set_xlabel('time')
        axs[4].legend()
        axs[4].grid(True)
        plt.show()

    def get_min_loss(self):
        return [self.model.P_i * self.f_path[i][1] for i in range(len(self.L_path))]
    
    def get_max_loss(self):
        return [self.model.P_e * self.f_path[i][1] for i in range(len(self.L_path))]

    def get_subtracted_loss(self, min_loss):
        return [self.L_path[i] - min_loss[i] for i in range(len(self.L_path))]