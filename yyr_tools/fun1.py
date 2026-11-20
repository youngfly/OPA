import matplotlib.pyplot as plt
import numpy as np
import math

a = 30
b = 30
x, y = np.mgrid[-a:a:200j, -b:b:200j]
z = abs(3*x-2*y)
fig=plt.figure(dpi=150)
ax = plt.axes(projection='3d')
ax.plot_surface(x,y,z, rstride=1, cstride=1, cmap= 'rainbow', alpha=0.9)


# todo **********************