import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
y = np.array([5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, .5])
for a, b in zip(x, y):
    plt.plot([a, 0], [0, b])
plt.show()
