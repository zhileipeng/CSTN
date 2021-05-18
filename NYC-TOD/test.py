import numpy as np
import pandas as pd

oddata = "../NYC-TOD/oddata.npy"
oddata = np.load(oddata, allow_pickle=True)[()]
data = oddata[0]
data_sum = np.zeros((75, 15, 5))
for i in range(520):
    data_sum += data[i]
data = np.zeros((15, 5))
for i in range(75):
    data += data_sum[i]

data = pd.DataFrame(data)
np.save("data_map.npy", data)

print(0)

