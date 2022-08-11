import numpy as np
import matplotlib.pyplot as plt
from main import *

if __name__ == '__main__':
    load_name = "model"

    grid_opt = load_optimizer(load_name)

    M = grid_opt.cost_to_go_m

    # Plot the cost to go matrix
    print('Enter a value for the consumption state:')
    c = int(input())
    if c not in np.arange(21):
        raise ValueError('Wrong input!')
    else:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        caxes = axes.matshow(M[:, c, :], cmap='RdYlGn_r')
        fig.colorbar(caxes)
        axes.set_xticks(grid_opt.model.B)
        axes.set_xticklabels(grid_opt.model.B, fontsize=12)
        axes.xaxis.set_ticks_position("bottom")
        axes.set_yticks(np.arange(len(grid_opt.model.O)))
        axes.set_yticklabels(grid_opt.model.O, fontsize=12)
        axes.set_xlabel('Battery state', fontsize=12)
        axes.set_ylabel('Output state', fontsize=12)
        plt.title('The costs to go for different Battery and Power\nplant states with consumption state ' + str(c),
                  fontsize=13)
        plt.show()

    # plot the markov chain transition matrix
    chain = grid_opt.model.chain
    fig = plt.figure()
    axes = fig.add_subplot(111)
    caxes = axes.matshow(chain.observed_p_matrix)
    fig.colorbar(caxes)
    plt.title('Transition matrix for the Markov chain,\ncalculated from real data')
    axes.set_xlabel('Consumption in the next state')
    axes.xaxis.set_ticks_position("bottom")
    axes.set_ylabel('Consumption in this state')
    plt.show()
