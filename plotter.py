import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')

"""This file is just for faster plotting in case of trying different layouts"""

X_embedded = np.load('X_embedded.npy')
y = np.load('y.npy')

# Plot all 3 components (3D)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(11.7,8.27))
scatter = ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], c=y, cmap='brg', marker='o', edgecolors='w')

ax.set(xlim=(-30, 30), ylim=(-20, 20), zlim=(-25, 25), 
        title='t-distributed Stochastic Neighbor Embedding (t-SNE)',
        xlabel='Component 1', ylabel='Component 2', zlabel='Component 3')

legend = ax.legend(*scatter.legend_elements(), loc='best', title='Quality index')

ax.add_artist(legend)

plt.show()

# # Plot a 2D projection
# fig, ax = plt.subplots(figsize=(11.7,8.27))

# scatter = ax.scatter(X_embedded[:,0], X_embedded[:,1], c=y, cmap='brg', marker='o', edgecolors='w') # Plotting a 2D projection

# ax.set(xlim=(-30, 30), ylim=(-20, 20), 
#         title='t-distributed Stochastic Neighbor Embedding (t-SNE)',
#         xlabel='Component 1', ylabel='Component 2')

# legend = ax.legend(*scatter.legend_elements(), loc='best', title='Quality index')

# ax.add_artist(legend)

# plt.show()