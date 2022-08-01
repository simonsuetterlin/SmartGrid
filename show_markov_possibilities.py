import numpy as np
from main import *
import mchmm as mc
import pickle
import matplotlib.pyplot as plt


#set name from which the cost to go matrix will be taken
load_name = "model"


grid_opt = load_optimizer(load_name)


chain = grid_opt.model.chain
fig = plt.figure()
axes = fig.add_subplot(111)
caxes = axes.matshow(chain.observed_p_matrix)
fig.colorbar(caxes)
plt.show()