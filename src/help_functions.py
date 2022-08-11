def produce_O_instant(x1):
    """
    Calculates the amount of produced electricity in the given time interval.

    Args:
        x1: state that defines the production of O for the interval
    """
    return x1[0] * 0.5


def overflow_O_instant(x1):
    """
    Calculates the amount of electricity that O produced more than is needed by the consumer.

    Args:
        x1: state that defines the production of O and consumer needs for the interval
    """
    return max(x1[0] - x1[1], 0) * 0.5


def deficit_O_instant(x1):
    """
    Calculates the amount of electricity the consumer needs more than got produced by O.

    Args:
        x1: state that defines the production of O and consumer needs for the interval
    """
    return max(x1[1] - x1[0], 0) * 0.5


def battery_usage_instant(x0, x1):
    """
    Calculates the amount of electricity used from the batery in the given time interval.

    Args:
        x0: defines the state at the start of the interval
        x1: defines the state at the end of the interval
    """
    return min(x0[2], deficit_O_instant(x1) * 0.5)
