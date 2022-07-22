import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import mchmm as mc


filename1 = "data/london_hourly/block_10.csv"
filename2 = "data/london_hourly/block_11.csv"
filename3 = "data/london_hourly/block_12.csv"
names = [filename1, filename2, filename3]

frame_types = dict(zip(['LCLid', 'tstp', 'energy(kWh/hh)'], [str, object, np.floating]))
kwargs = {
    'dtype': frame_types,
    'na_values': 'Null',
    'usecols': ['energy(kWh/hh)',]
}
dataframes = (pd.read_csv(name, **kwargs) for name in names)


def init_chain(max_output, sample_size=1000):
    data = pd.concat(dataframes, ignore_index=True)
    data.dropna(inplace=True)
    data_numeric= pd.to_numeric(data['energy(kWh/hh)'][:sample_size])
    data_numeric = data_numeric * (max_output+1) / np.max(data_numeric)
    np.rint(data_numeric, out=data_numeric)

    chain = mc.MarkovChain().from_data(data_numeric)
    return chain