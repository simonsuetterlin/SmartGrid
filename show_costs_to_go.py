import pickle
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets, interactive
from main import *



#set name from which the cost to go matrix will be taken
load_name = "model"


grid_opt = load_optimizer(load_name)

M = grid_opt.cost_to_go_m



#fig, axs = plt.subplots(4, 5, dpi=100, figsize=(20, 12))
#for i in range(M.shape[1]):
#    axs[i].step(M[:,i,:])

#show the costs to go for the different states. The consumption states are distributed over the different figures
#

while True:
    print('Enter a value for the consumption state:')
    c = int(input())
    if c not in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
        print('Wrong input!')
        break
    else:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        caxes = axes.matshow(M[:,c,:])
        fig.colorbar(caxes)
        axes.set_xticks(grid_opt.model.B)
        axes.set_xticklabels(grid_opt.model.B)
        axes.set_yticks(np.arange(len(grid_opt.model.O)))
        axes.set_yticklabels(grid_opt.model.O)
        axes.set_xlabel('Battery state')
        axes.set_ylabel('Output state')
        plt.title('The costs to go for different Battery and Power\nplant states with consumption state '+str(c))
        plt.show()


for i in [0,1]:#grid_opt.model.V:
    fig = plt.figure()
    axes = fig.add_subplot(111)
    caxes = axes.matshow(M[:,0,:])
    fig.colorbar(caxes)
    axes.set_xticks(grid_opt.model.B)
    axes.set_xticklabels(grid_opt.model.B)
    axes.set_yticks(np.arange(len(grid_opt.model.O)))
    axes.set_yticklabels(grid_opt.model.O)
    axes.set_xlabel('Battery state')
    axes.set_ylabel('Output state')
    plt.title('The costs to go for different Battery and Power\nplant states with consumption state '+str(i))


plt.show()








