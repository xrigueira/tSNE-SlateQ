import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from matplotlib import rcParams
rcParams['font.family'] = 'monospace'

"""This file is just for faster plotting in case of trying different layouts"""

X_embedded = np.load('X_embedded.npy')
y = np.load('y.npy')

# Plot all 3 components (3D)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(11.7,8.27))
scatter = ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], c=y, cmap='brg', marker='o', edgecolors='w')

ax.set(xlim=(-30, 30), ylim=(-20, 20), zlim=(-25, 25))

ax.set_title('(a)', fontname='Arial', fontsize=16)
ax.set_xlabel('Component 1', fontname='Arial', fontsize=14)
ax.set_ylabel('Component 2', fontname='Arial', fontsize=14)
ax.set_zlabel('Component 3', fontname='Arial', fontsize=14)

legend = ax.legend(*scatter.legend_elements(), loc='best', title='Quality index', fontsize=14, title_fontsize='13', prop={'family':'Arial'})
plt.setp(legend.get_title(), fontname='Arial')

ax.add_artist(legend)

plt.show()

# Plot first 2D projection
fig, ax = plt.subplots(figsize=(11.7,8.27))

scatter = ax.scatter(X_embedded[:,0], X_embedded[:,1], c=y, cmap='brg', marker='o', edgecolors='w') # Plotting a 2D projection

ax.set(xlim=(-30, 30), ylim=(-20, 20))

ax.set_title('(a)', fontname='Arial', fontsize=16)
ax.set_xlabel('Component 1', fontname='Arial', fontsize=14)
ax.set_ylabel('Component 2', fontname='Arial', fontsize=14)

legend = ax.legend(*scatter.legend_elements(), loc='best', title='Quality index', fontsize=14, title_fontsize='13', prop={'family':'Arial'})
plt.setp(legend.get_title(), fontname='Arial')

ax.add_artist(legend)

# plt.show()

plt.savefig('Figures/figure1a.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Plot second 2D projection
fig, ax = plt.subplots(figsize=(11.7,8.27))

scatter = ax.scatter(X_embedded[:,0], X_embedded[:,2], c=y, cmap='brg', marker='o', edgecolors='w') # Plotting a 2D projection

ax.set(xlim=(-30, 30), ylim=(-25, 25))

ax.set_title('(b)', fontname='Arial', fontsize=16)
ax.set_xlabel('Component 1', fontname='Arial', fontsize=14)
ax.set_ylabel('Component 3', fontname='Arial', fontsize=14)

legend = ax.legend(*scatter.legend_elements(), loc='best', title='Quality index', fontsize=14, title_fontsize='13', prop={'family':'Arial'})
plt.setp(legend.get_title(), fontname='Arial')

ax.add_artist(legend)

# plt.show()

plt.savefig('Figures/figure1b.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Plot third 2D projection
fig, ax = plt.subplots(figsize=(11.7,8.27))

scatter = ax.scatter(X_embedded[:,1], X_embedded[:,2], c=y, cmap='brg', marker='o', edgecolors='w') # Plotting a 2D projection

ax.set(xlim=(-20, 20), ylim=(-25, 25))

ax.set_title('(c)', fontname='Arial', fontsize=16)
ax.set_xlabel('Component 2', fontname='Arial', fontsize=14)
ax.set_ylabel('Component 3', fontname='Arial', fontsize=14)

legend = ax.legend(*scatter.legend_elements(), loc='best', title='Quality index', fontsize=14, title_fontsize='13', prop={'family':'Arial'})
plt.setp(legend.get_title(), fontname='Arial')

ax.add_artist(legend)

# plt.show()

plt.savefig('Figures/figure1c.pdf', format='pdf', dpi=300, bbox_inches='tight')