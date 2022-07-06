from rk4step import rk4step
import numpy as np
import casadi as ca
import random

"""
Ts = .5
N = 19
x0 = np.array([1, 0])
t_grid = np.linspace(0, Ts*N, N+1)
t_span = [0, Ts*N]
t0 = 0


X_rk4 = np.zeros((x0.size, N+1))
X_rk4[:,0] = x0
for k in range(1, len(t_grid)):
    X_rk4[:,k] = rk4step(f, Ts,  X_rk4[:,k-1], t0).full().flatten()
"""


P_e = 20
P_i = 10
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
        return 0
    elif(o0 < v and o1 < v):
        return 0.5*(v - o0 + v - o1) * P_e
    else:
        t = (v - o0)/(o1-o0)
        if(o0 < v and o1 >= v):
            return 0.5 * (o1 - v) * (1-t) * P_e
        elif(o0 >= v and o1 < v):
            return 0.5 * (o0 - v) * t * P_e 

def L_i(o0, o1):
    return 0.5 * (o0 + o1) * P_i


"""
def V(v):
    return v * C

def O_i(k, O_i_prev, u):
    return O_i_prev + u * k

def L_i(k, O_i_prev, u):
    return P_i * O_i(k, O_i_prev, u)

def O_e(k, O_i_prev, u, v):
    return np.max([V(v) - O_i(k, O_i_prev, u), 0])

def L_e(k, O_i_prev, u, v):
    return P_e * O_e(k, O_i_prev, u, v)

# integrator wrong
def L(x0, u, v):
    def f(k, v):
        k = ca.SX.sym('k')
        return L_e(k, x0[0], u, v) + L_i(k, x0[0], u)

    x = ca.SX.sym('x')
    ode = {'x':x, 'p':v, 't':k,'ode':f}

    F = ca.integrator('F', 'idas', ode,{'t0':0,'tf':1})
    r = F(x0=0)
    return(r['xf'])
"""


depth = 10
dim_O = 5
dim_V = 5

U = [-2, -1,0,1,2]
O_range = list(range(dim_O))
V_range = list(range(dim_V))
V_realisation = [4,4,4,4,4,1,0,4,2,3,3] #V_realisation = C = np.array([random.randint(min(V_range),max(V_range)) for j in range(depth+1)])

def calculate_cost_to_go_matrix_sequence(V_realisation):
    M = np.zeros((depth,dim_O,dim_V), dtype=float)
    for i in range(depth):
        if(i != 0):
            M[i,:,:] = calculate_cost_to_go_matrix(V_realisation[i], M[i-1,:,:])
        else:
            M[0,:,:] = calculate_cost_to_go_matrix_final_step()

    return M[-1,:,:]

# calculate matrix with includes indexes for actions U that are optimal for given x0 in matrix
def calculate_optimal_step_matrix(V_realisation):
    M_cost_to_go = calculate_cost_to_go_matrix_sequence(V_realisation[1:])
    optimal_steps = np.zeros((dim_O,dim_V), dtype=int)
    for o_index in range(dim_O):
        for v_index in range(dim_V):
            step_cost_to_go_array = np.zeros(len(U), dtype=float)
            for u_index in range(len(U)):
                x0 = (O_range[o_index], V_range[v_index])
                step_cost_to_go_array[u_index] = calculate_path_cost(x0, U[u_index], V_realisation[0], M_cost_to_go)
            min_index_u = 0
            for i in range(1, len(step_cost_to_go_array)):
                if (step_cost_to_go_array[i] < step_cost_to_go_array[min_index_u]):
                    min_index_u = i
            optimal_steps[o_index, v_index] = U[min_index_u]
    return optimal_steps

    # TODO change end score
def calculate_cost_to_go_matrix_final_step():
    M = np.ndarray((dim_O, dim_V), dtype=float)
    for o_index in range(dim_O):
        for v_index in range(dim_V):
            M[o_index,v_index] = 0 # end costs are zeros at the moment
    return M

# TODO optimieren mit list comprehension?
def calculate_cost_to_go_matrix(v_new, M_N_plus_1):
    M = np.ndarray((dim_O, dim_V), dtype=float)
    for o_index in range(dim_O):
        for v_index in range(dim_V):
            M[o_index,v_index] = cost_to_go((O_range[o_index], V_range[v_index]), v_new, M_N_plus_1)
    return M
    # [[cost_to_go((o,v), U, M_N_plus_1) for o in range(dim_O)] for v in range(dim_V)]


def cost_to_go(x0, v_new, M_N_plus_1):
    step_cost_to_go_array = np.zeros(len(U), dtype=float)
    for u_index in range(0,len(U)):
        step_cost_to_go_array[u_index] = calculate_path_cost(x0, U[u_index], v_new, M_N_plus_1)
    return np.min(step_cost_to_go_array)

#TODO ERWARTUNGSWERT
def calculate_path_cost(x0, u, v_new, M_N_plus_1):
    if(x0[0] + u in O_range):
        stage_cost = L(x0, u, v_new)
        prev_cost_to_go = M_N_plus_1[f(x0, u, v_new)]
        return stage_cost + prev_cost_to_go
    else:
        return np.inf

opt_step = calculate_optimal_step_matrix(V_realisation)
print(opt_step)