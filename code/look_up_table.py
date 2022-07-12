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
            np.zeros(dim, dtype=np.int8), axes=transposition) + np.arange(dim[i], dtype=np.int8), axes=transposition)
        ls.append(np.reshape(x, newshape=size))
    matr_index[:] = list(zip(*ls))
    return np.reshape(matr_index, newshape=dim)