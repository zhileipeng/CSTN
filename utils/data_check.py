import numpy as np

data = np.load("../NYC-TOD/hk_ODData.npy", allow_pickle=True)[()]
print(data.max())
data = data.reshape((-1, 50, 10, 5))
np.save("hk_ODData2.npy", data)
print(data.shape)
