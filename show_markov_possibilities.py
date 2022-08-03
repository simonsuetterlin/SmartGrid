import numpy as np
from main import *
import mchmm as mc
import pickle
import matplotlib.pyplot as plt


#set name from which the cost to go matrix will be taken
load_name = "model"


grid_opt = load_optimizer(load_name)


chain = grid_opt.model.chain
print(chain.observed_p_matrix)
fig = plt.figure()
axes = fig.add_subplot(111)
caxes = axes.matshow(chain.observed_p_matrix, cmap = 'magma')
fig.colorbar(caxes)
plt.title('Transition matrix for the Markov chain, calculated from real data')
axes.set_xlabel('Consumption in the next state')
axes.set_ylabel('Consumption in this state')
plt.show()