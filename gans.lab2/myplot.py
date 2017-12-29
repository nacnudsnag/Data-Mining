                                                 
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

fig = plt.figure()
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

# use this to write to a file; look at the file with display
#fig.savefig('my_figure.png')                                                    

# works at Bowdoin; didn't work from off campus
plt.show()
