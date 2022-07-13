import numpy as np

def look_up_table(dim):
    """
    Calculates a look up table.
    Each entree of this matrix is exactly the tuple
    with which it can be accesed by.
    
    Args:
        dim (int, tuple): Dimensions of the matrix which
    
    Returns:
        Matrix with shape dim and entrees tuple of length len(dim)
    """
    assert isinstance(dim, tuple), (f"Wrong input type of variable dimension.\n"
                                           f" Needs type tuple, but {type(dim)} was given")
    n = len(dim)
    size = np.prod(dim)
    ls = []
    matr_index = np.empty(size, dtype=tuple)
    for i in range(n):
        transposition = list(range(n))
        transposition[i], transposition[n-1] = n-1, i
        x = np.transpose(np.transpose(
            np.zeros(dim, dtype=np.int8), axes=transposition) 
                         + np.arange(dim[i], dtype=np.int8),
                         axes=transposition)
        ls.append(np.reshape(x, newshape=size))
    matr_index[:] = list(zip(*ls))
    return np.reshape(matr_index, newshape=dim)

def convert_index_to_state(index, state_space):
    """
    Converts an multi-index to a state of given state space.
    
    Args:
        index (tuple): Multi Index
        state_space (list of lists): 
    
    Returns:
        tuple: a state
    """
    assert len(index) == len(state_space),(
        f"The length of given index {len(index)} does not match the dimension "
        f"of the state space {len(state_space)}."
    )
    for i, ind in enumerate(index):
        assert 0 <= ind < len(state_space[i]),(
            f"The {i}-th index is out of bounds. "
            f"{ind} needs to be between 0 and {len(state_space[i])}"
        )
    return tuple([state_space[i][ind] for i, ind in enumerate(index)])

def convert_state_to_index(state, state_space):
    """
    Converts an multi-index to a state of given state space.
    
    Args:
        state (tuple): Multi Index
        state_space (list of lists): 
    
    Returns:
        tuple: a multi-index
    """
    assert len(state) == len(state_space),(
        f"The length of given index {len(state)} does not match the dimension "
        f"of the state space {len(state_space)}."
    )
    return tuple([np.where(state_space[i] == value)[0][0] for i, value in enumerate(state)])