import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))

plt.show()

for i in range(0, 10000):
	continue
plt.close()

plt.plot(x, np.cos(x))
plt.close()

plt.show()
