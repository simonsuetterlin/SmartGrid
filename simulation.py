import random
import numpy as np
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self, model, optimal_step_matrix):
        self.model = model
        self.optimal_step_matrix = optimal_step_matrix
        self.f_path = []
        self.L_path = []
        
    def simulate(self, T, O_start=0, V_realisation=None):
        # if no realisation of V is given it gets calculated randomly
        if V_realisation == None:
            V_realisation = self.model.distribution.rvs(size=T)
            
        x0 = (O_start, random.randint(min(self.model.V), max(self.model.V)))
        for i in range(T):
            u = self.optimal_step_matrix[x0[0], x0[1]]
            v = V_realisation[i]
            x1 = self.model.f(x0, u, v)
            self.f_path.append(self.model.f(x0, u, v))
            self.L_path.append(self.model.L(x0, x1))
            x0 = x1
            
    def plot_path(self):
        fig, axs = plt.subplots(2, 1, figsize=(8,5), dpi=150)
        t = np.arange(0, len(self.L_path))
        axs[0].plot(t, [i[0] for i in self.f_path], label="O")
        axs[0].plot(t, [i[1] for i in self.f_path], label="V")
        axs[0].set_ylabel('O and V')
        axs[0].legend()
        axs[0].grid(True)
        axs[1].plot(self.L_path, label="loss")
        axs[1].plot(self.get_min_loss(), label="min loss")
        axs[1].plot(self.get_max_loss(), label="max loss")
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('t')
        axs[1].legend()
        axs[1].grid(True)
        plt.show()

    def get_min_loss(self):
        return [self.model.P_i * self.f_path[i][1] for i in range(len(self.L_path))]
    
    def get_max_loss(self):
        return [self.model.P_e * self.f_path[i][1] for i in range(len(self.L_path))]