
import numpy as np
import pandas as pd

from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.manifold._t_sne import _joint_probabilities

from matplotlib import pyplot as plt
plt.style.use('ggplot')

"""t-SNE implementation on slate quality data"""

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
def _gradient_descent(obj_func, p0, args, it=0, n_iter=100, n_iter_check=1, n_iter_without_progress=300, momentum=0.8, learning_rate=200.0, min_gain=0.01, min_grad_norm=1e-7):

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

if __name__ == '__main__':

    # Read the data and drop the expendable columns
    df = pd.read_csv('Data/PasCan.csv', delimiter=';')
    df = df.drop(columns= ['Sondeo', 'X', 'Y', 'Z', 'Profundidad', 'RQD'])

    # Create a 1D array with the target values and drop the column with this information
    y = df['Calidad'].to_numpy(dtype=float)
    df = df.drop(columns=['Calidad'])
    
    # Create a 2D numpy array with the same shape as the df
    X = np.empty((len(df), df.shape[1]), dtype=float)
    
    # Iterate over the rows of the df and assign them to the corresponding items in the 2D numpy.ndarray
    for i, row in df.iterrows():
        X[i,:] = row.values

    # Define machine epsilon, number of components, and perplexity
    MACHINE_EPSILON = np.finfo(np.double).eps
    n_components = 3
    perplexity = 45

    #------------------------***------------------------
    # # Call the fit function
    # X_embedded = fit(X)
    
    # # Save the result for further processing
    # np.save('X_embedded.npy', X_embedded, allow_pickle=False, fix_imports=False)
    # np.save('y.npy', y, allow_pickle=False, fix_imports=False)

    # # Plot the results
    # # Plot all 3 components (3D)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(11.7,8.27))
    
    # scatter = ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], c=y, cmap='brg', marker='o', edgecolors='w')
    
    # ax.set(xlim=(-30, 30), ylim=(-20, 20), zlim=(0, 30), 
    #     title='t-distributed Stochastic Neighbor Embedding (t-SNE)',
    #     xlabel='Component 1', ylabel='Component 2', zlabel='Component 3')
    
    # legend = ax.legend(*scatter.legend_elements(), loc='best', title='Quality index')

    # ax.add_artist(legend)
    
    # plt.show()
    
    # # Plot a 2D projection
    # fig, ax = plt.subplots(figsize=(11.7,8.27))

    # scatter = ax.scatter(X_embedded[:,0], X_embedded[:,1], c=y, cmap='brg', marker='o', edgecolors='w') # Plotting a 2D projection

    # ax.set(xlim=(-30, 30), ylim=(-20, 20), 
    #         title='t-distributed Stochastic Neighbor Embedding (t-SNE)',
    #         xlabel='Component 1', ylabel='Component 2')

    # legend = ax.legend(*scatter.legend_elements(), loc='best', title='Quality index')

    # ax.add_artist(legend)
    
    # plt.show()
    #------------------------***------------------------
    
    # Same method but implemented with scikit-learn
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000, verbose=1, random_state=123)
    X_embedded = tsne.fit_transform(X)
    
    # Save the result for further processing
    np.save('X_embedded.npy', X_embedded, allow_pickle=False, fix_imports=False)
    np.save('y.npy', y, allow_pickle=False, fix_imports=False)
    
    # Plot all 3 components (3D)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(11.7,8.27))
    # scatter = ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], c=y, cmap='brg', marker='o', edgecolors='w')
    
    # ax.set(xlim=(-30, 30), ylim=(-20, 20), zlim=(0, 30), 
    #     title='t-distributed Stochastic Neighbor Embedding (t-SNE)',
    #     xlabel='Component 1', ylabel='Component 2', zlabel='Component 3')
    
    # legend = ax.legend(*scatter.legend_elements(), loc='best', title='Quality index')

    # ax.add_artist(legend)
    
    # plt.show()
    
    # Plot a 2D projection
    fig, ax = plt.subplots(figsize=(11.7,8.27))

    scatter = ax.scatter(X_embedded[:,0], X_embedded[:,1], c=y, cmap='brg', marker='o', edgecolors='w') # Plotting a 2D projection

    ax.set(xlim=(-30, 30), ylim=(-20, 20), 
            title='t-distributed Stochastic Neighbor Embedding (t-SNE)',
            xlabel='Component 1', ylabel='Component 2')

    legend = ax.legend(*scatter.legend_elements(), loc='best', title='Quality index')

    ax.add_artist(legend)
        
    plt.show()