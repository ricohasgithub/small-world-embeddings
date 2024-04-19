
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def embedding(point):
    return -np.sqrt(0.3*(point[0]**2 + point[1]**2) + 1) + 0.05

def approximate_geodesic(p1, p2, steps=50):
    """Generate points between p1 and p2 on the hyperboloid."""
    p1, p2 = np.asarray(p1), np.asarray(p2)
    t = np.linspace(0, 1, steps)
    line = np.vstack([(1-t)*p1 + t*p2 for t in t])
    # Filter points to only include those on the hyperboloid
    return line[np.abs(line[:, 0]**2 + line[:, 1]**2 - line[:, 2]**2 - 1) < 0.1]

X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)

Z = np.sqrt(0.3*(X **2 + Y **2) + 1)

V = [[0, 0], [2, 2], [-2*np.sqrt(2), 0], [2, -2]]
embedded = [[point[0], point[1], embedding(point)] for point in V]

# Create a figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.6, vmin=-5, vmax=5)
ax.plot_surface(X, Y, -Z, cmap='jet', alpha=0.6, vmin=-5, vmax=5)

for point in embedded:
    ax.scatter(point[0], point[1], point[2], color='black', s=50)

# Connect points with geodesic approximations
for i, p1 in enumerate(embedded):
    for j, p2 in enumerate(embedded):
        if i < j:  # Connect each pair only once
            line = approximate_geodesic(p1, p2)
            print(line)
            projected = embedding(line)
            ax.plot(projected[:, 0], projected[:, 1], projected[:, 2], 'green')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# draw sphere
u, v = np.mgrid[0:2*np.pi:25j, 0:np.pi:25j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)

# Create a figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# alpha controls opacity
ax.plot_surface(x, y, z, cmap='jet', alpha=0.6)
ax.scatter(0, -1, 0, color='black')
ax.scatter(0, 1, 0, color='black')
ax.scatter(1, 0, 0, color='black')
ax.scatter(-1, 0, 0, color='black')

plt.show()
