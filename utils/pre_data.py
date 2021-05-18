import numpy as np
from numpy import newaxis
from utils.dataset import load_data


x, y, w, m = load_data(391, 5)  # 241
#
splitPoint = int(x.shape[0] * 0.8)

x_train = x[: splitPoint]
y_train = y[: splitPoint]
w_train = w[: splitPoint]
m_train = m[: splitPoint]
x_val = x[splitPoint:]
y_val = y[splitPoint:]
w_val = w[splitPoint:]
m_val = m[splitPoint:]


np.save("./utils/train_data.npy", x_train)
np.save("./utils/train_label.npy", y_train)
np.save("./utils/train_weather.npy", w_train)
np.save("./utils/train_meta.npy", m_train)
np.save("./utils/val_data.npy", x_val)
np.save("./utils/val_label.npy", y_val)
np.save("./utils/val_weather.npy", w_val)
np.save("./utils/val_meta.npy", m_val)
print("Done.")
