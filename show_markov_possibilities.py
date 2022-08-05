import numpy as np
from main import *
import mchmm as mc
import pickle
import matplotlib.pyplot as plt


#set name from which the cost to go matrix will be taken
load_name = "model"


grid_opt = load_optimizer(load_name)


def array_to_matrix(array):
    begin = '\\begin{pmatrix} \n'
    data = ''
    for line in array:        
        if line.size == 1:
            data = data + ' %.3f &'%line
            data = data + r' \\'
            data = data + '\n'
            continue
        for element in line:
            data = data + ' %.3f &'%element

        data = data + r' \\'
        data = data + '\n'
    end = '\end{pmatrix}'
    print(begin + data + end)

chain = grid_opt.model.chain
#array_to_matrix(chain.observed_p_matrix)
fig = plt.figure()
axes = fig.add_subplot(111)
caxes = axes.matshow(chain.observed_p_matrix)
fig.colorbar(caxes)
plt.title('Transition matrix for the Markov chain,\ncalculated from real data')
axes.set_xlabel('Consumption in the next state')
axes.xaxis.set_ticks_position("bottom")
axes.set_ylabel('Consumption in this state')
plt.show()