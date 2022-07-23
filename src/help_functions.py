import numpy as np

def produce_O_instant(x0, x1):
    return x1[0]

def overflow_O_instant(x0, x1):
    return max(x1[0] - x1[1], 0)

def deficit_O_instant(x0, x1):
    return max(x1[1] - x1[0], 0)

def battery_usage_instant(x0, x1):
    return min(x0[2], deficit_O_instant(x0, x1))


def produce_O(x0, x1):
    return 0.5 * (x0[0] + x1[0])

def overflow_O(x0, x1):
    o0 = x0[0]
    o1, v1 = x1[:2]
    if (o0 > v1 and o1 > v1):
        return [0., 0.5 * (o0 + o1 - 2 * v1)]
    elif (o0 <= v1 and o1 <= v1):
        return [0., 0.]
    else:
        t = (v1 - o0) / (o1 - o0)
        if (o0 <= v1 and o1 > v1):
            return [0., 0.5 * (o1 - v1) * (1 - t)]
        elif (o0 > v1 and o1 <= v1):
            return [0.5 * (o0 - v1) * t, 0.]


def deficit_O(x0, x1):
    return x1[1] + np.sum(overflow_O(x0, x1)) - produce_O(x0, x1)


def battery_usage(x0, x1, max_charge):
    deficit = deficit_O(x0, x1)
    overflow = overflow_O(x0, x1)
    return min(x0[2] + overflow[0], max_charge, deficit)
