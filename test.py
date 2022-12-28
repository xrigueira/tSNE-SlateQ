
"""Example 1: 
https://www.datatechnotes.com/2020/11/tsne-visualization-example-in-python.html"""

# import pandas as pd
# import seaborn as sns
# from numpy import reshape
# from sklearn.manifold import TSNE
# from keras.datasets import mnist
# from matplotlib import pyplot as plt
# from sklearn.datasets import load_iris

# # Iris dataset t-SNE fitting and visualizing
# iris = load_iris()
# X = iris.data
# y = iris.target

# # Define the t-SNE model
# tsne = TSNE(n_components=2, perplexity=10, verbose=1, random_state=123)
# z = tsne.fit_transform(X)

# # Visualize the results
# df = pd.DataFrame()
# df['y'] = y
# df['comp-1'] = z[:,0]
# df['comp-2'] = z[:,1]

# sns.scatterplot(x='comp-1', y='comp-2', hue=df.y.tolist(), palette=sns.color_palette('hls', 3), data=df).set(title='Iris data t-SNE projection')
# plt.show()

# # MNIST dataset t-SNE fitting and visualizing
# (x_train, y_train), (_ , _) = mnist.load_data()
# x_train = x_train[:3000]
# y_train = y_train[:3000]

# # Reshape into two dimensions
# x_mnist = reshape(x_train, [x_train.shape[0], x_train.shape[1] * x_train.shape[2]])

# # Define the t-SNE model
# tsne = TSNE(n_components=2, verbose=1, random_state=123)
# z = tsne.fit_transform(x_mnist)

# # Visualize the results
# df = pd.DataFrame()
# df["y"] = y_train
# df["comp-1"] = z[:,0]
# df["comp-2"] = z[:,1]

# sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 10), data=df).set(title="MNIST data T-SNE projection")
# plt.show()

"""Example 2: Full implementation without scikit plus scikit 
https://towardsdatascience.com/t-sne-python-example-1ded9953f26
"""

import numpy as np
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances
from sklearn.manifold._t_sne import _joint_probabilities

from matplotlib import pyplot as plt

# Load data
X, y = load_digits(return_X_y=True)
print(type(X), len(X))
print(type(y), len(y))
print(len(X[1]))
print(X)
print(y)

# Define machine epsilon, number of components and perplexity
MACHINE_EPSILON = np.finfo(np.double).eps
n_components = 2
perplexity = 30

# Define the fit function
def fit(X):
    
    n_samples = X.shape[0]
    
    # Calculate Euclidean distance
    distances = pairwise_distances(X, metric='euclidean', squared=True)
    
    # Calculate the joint probabilities p_ij from the distances
    P = _joint_probabilities(distances=distances, desired_perplexity=perplexity, verbose=False)
    
    # The embedding is initialized with iid samples from Gaussian distributions with std = 1e-4
    X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, n_components).astype(np.float32)
    
    # Define degrees of freedom according to VanDerMaaten2008
    degrees_of_freedom = max(n_components - 1, 1)
    
    return _tsne(P, degrees_of_freedom, n_samples, X_embedded=X_embedded)

# Define the t-SNE function
def _tsne(P, degrees_of_freedom, n_samples, X_embedded):
    
    params = X_embedded.ravel() # Flatten array
    
    obj_func = _kl_divergence
    
    params = _gradient_descent(obj_func, params, [P, degrees_of_freedom, n_samples, n_components])
    
    X_embedded = params.reshape(n_samples, n_components)
    
    return X_embedded

# Compute the error in the form of the Kullbacl-Leibler divergence and the gradient descent
def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components):
    
    X_embedded = params.reshape(n_samples, n_components)
    
    # Define the Student's (Cauchy) distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1
    dist **= (degrees_of_freedom + 1.0) / (-2.0)
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
    
    # Kullback-Liebler divergence of P abd Q
    kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    
    # Compute gradient: dC/dY
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    
    for i in range(n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'), X_embedded[i] - X_embedded)
    
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    
    return kl_divergence, grad

# Define gradient descent
def _gradient_descent(obj_func, p0, args, it=0, n_iter=1000, n_iter_check=1, n_iter_without_progress=300, momentum=0.8, learning_rate=200.0, min_gain=0.01, min_grad_norm=1e-7):

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it
    
    for i in range(it, n_iter):
        
        error, grad = obj_func(p, *args)
        
        grad_norm = linalg.norm(grad)
        
        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update
        
        print("[t-SNE] Iteration %d: error = %.7f," " gradient norm = %.7f"% (i + 1, error, grad_norm))
        
        if error < best_error:
            best_error = error
            best_iter = 1
        
        elif i - best_iter > n_iter_without_progress:
            break
        
        if grad_norm <= min_grad_norm:
            break
        
    return p

# # Call the fit function
# X_embedded = fit(X)

# # Plot the results
# sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)
# plt.show()

# # Same method but implemented with scikit-learn
# tsne = TSNE()
# X_embedded = tsne.fit_transform(X)

# sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)
# plt.show()