import numpy as np


def produce_O(o):
    return o


def overflow_O(o, v):
    if o > v:
        return o - v
    else:
        return 0


def deficit_O(o, v):
    if o > v:
        return 0
    else:
        return v - o


def battery_usage(o, v, b_level):
    deficit = deficit_O(o, v)
    return max(min(deficit, b_level), 0)
