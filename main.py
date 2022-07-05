from rk4step import rk4step
import numpy as np
import casadi as ca

Ts = .5
N = 19
x0 = np.array([1, 0])
t_grid = np.linspace(0, Ts*N, N+1)
t_span = [0, Ts*N]
t0 = 0

def f(t0, x):
    return ca.vertcat(x[1], -.2*x[1] - x[0])

X_rk4 = np.zeros((x0.size, N+1))
X_rk4[:,0] = x0
for k in range(1, len(t_grid)):
    X_rk4[:,k] = rk4step(f, Ts,  X_rk4[:,k-1], t0).full().flatten()


P_e = 10
P_i = 5
C = 50
M = 5

def L(x0, u, v):
    x1 = f(x0, u, v)
    return L_i(x0[0], x1[0]) + L_e(x0[0], x1[0], v)

def f(x0, u, v):
    o1 = x0[0] + u
    v1 = v 
    return (o1, v1)

def L_e(o0, o1, v):
    if(o0 >= v and o1 >= v):
        return 0.5 * np.abs(o0 - o1) * P_e
    elif(o0 < v and o1 < v):
        return 0
    else:
        t = (v - o0)/(o1-o0)
        if(o0 < v and o1 >= v):
            return 0.5 * (o1 - v) * (1-t) * P_e
        elif(o0 >= v and o1 < v):
            return 0.5 * (o0 - v) * t * P_e 

def L_i(o0, o1):
    return 0.5 * np.abs(o0 - o1) * P_i



depth = 10
dim_O = 5
dim_V = 5

U = [5,10,15,20]
O = [10,20,30,40]
V = [50,60,70,80,90,100]

def calculate_cost_to_go_final_step():
    M = np.zeros((dim_O,dim_V), dtype=float)
    return M

def calculate_cost_to_go(M_N_plus_1):
    if(M_N_plus_1[0,0] == 0): # nur platzfüller
        M = np.ones((dim_O,dim_V), dtype=float)
    else:
        M = np.zeros((dim_O,dim_V), dtype=float)
    return M

def calculate_cost_to_go_matrix_sequence():
    M = np.zeros((depth,dim_O,dim_V), dtype=float)
    for i in range(depth):
        if(i != 0):
            M[i,:,:] = calculate_cost_to_go_matrix(M[i-1,:,:])
        else:
            M[0,:,:] = calculate_cost_to_go_matrix_final_step()

    return M[-1,:,:]

# calculate matrix with includes indexes for actions U that are optimal for given x0 in matrix
def calculate_optimal_step_matrix():
    M_cost_to_go = calculate_cost_to_go_matrix_sequence()
    optimal_steps = np.zeros((dim_O,dim_V), dtype=int)
    for o_index in range(dim_O):
        for v_index in range(dim_V):
            step_cost_to_go_array = np.zeros(len(U), dtype=float)
            for u_index in range(len(U)):
                x0 = (O[o_index], V[v_index])
                step_cost_to_go_array[u_index] = calculate_path_cost(x0, U[u_index], M_cost_to_go)
            min_index_u = 0
            for i in range(2, len(step_cost_to_go_array)):
                if (step_cost_to_go_array[i] < step_cost_to_go_array[min_index_u]):
                    min_index_u = i
            optimal_steps[o_index, v_index] = min_index_u
    return optimal_steps

    # TODO change end score
def calculate_cost_to_go_matrix_final_step():
    M = np.ndarray((dim_O, dim_V), dtype=float)
    for o_index in range(dim_O):
        for v_index in range(dim_V):
            M[o_index,v_index] = 0 # end costs are zeros at the moment

# TODO optimieren mit list comprehension?
def calculate_cost_to_go_matrix(M_N_plus_1):
    M = np.ndarray((dim_O, dim_V), dtype=float)
    for o_index in range(dim_O):
        for v_index in range(dim_V):
            M[o_index,v_index] = cost_to_go((O[o_index], V[v_index]), U, M_N_plus_1)
    return M
    # [[cost_to_go((o,v), U, M_N_plus_1) for o in range(dim_O)] for v in range(dim_V)]


def cost_to_go(x0, M_N_plus_1):
    step_cost_to_go_array = np.zeros(len(U), dtype=float)
    for u_index in range(0,len(U)):
        step_cost_to_go_array[u_index] = calculate_path_cost(x0, U[u_index], M_N_plus_1)
    return np.min(step_cost_to_go_array)

#TODO ERWARTUNGSWERT
def calculate_path_cost(x0, u, M_N_plus_1):
    stage_cost = L(x0, u)
    prev_cost_to_go = M_N_plus_1[f(x0, u)]
    return stage_cost + prev_cost_to_go


opt_step = calculate_optimal_step_matrix()