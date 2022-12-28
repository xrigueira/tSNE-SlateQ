
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 3)

from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances
from sklearn.manifold._t_sne import _joint_probabilities

from matplotlib import pyplot as plt

"""t-SNE implementation on slate quality data
https://towardsdatascience.com/t-sne-python-example-1ded9953f26"""

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

    # My case is two dimensional. MNIST is 3-dimensional because it works with 28x28 images. Son 60k images (matrices)
    # My case is like the iris dataset
    # I would have to load the csv data in two variables X and y, both <numpy.ndarray>. Each row of the database as one
    # item in the X 2D numpy.ndarray. Except "Calidad" which would be loaded in y as a 1D numpy.ndarray

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
    perplexity = 30

    # # Call the fit function
    # X_embedded = fit(X)

    # # Plot the results
    # sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)
    # plt.show()

    # Same method but implemented with scikit-learn
    tsne = TSNE(n_components=n_components, perplexity=perplexity, verbose=1, random_state=123)
    X_embedded = tsne.fit_transform(X)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], c=y)
    
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)
    
    plt.show()