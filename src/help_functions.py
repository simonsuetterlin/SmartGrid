def produce_O_instant(x0, x1):
    return x1[0] * 0.5

def overflow_O_instant(x0, x1):
    return max(x1[0] - x1[1], 0) * 0.5

def deficit_O_instant(x0, x1):
    return max(x1[1] - x1[0], 0) * 0.5

def battery_usage_instant(x0, x1):
    return min(x0[2], deficit_O_instant(x0, x1) * 0.5)

