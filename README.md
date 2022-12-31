# tSNE-SlateQ

t-distributed Stochastic Neighbour Embedding is a nonlinear statistical method for visualizing high-dimensional data in 2 or 3 dimensions. In other words, is a manifold dimensional reduction method that allows the representation of data with an important number of features through data points in a low-dimensional space while preserving maximum information. It is based on Stochastic Neighbour Embedding originally developed by Hinton and Roweis [1], while the t-SNE introduces a t-distributed variant as proposed by van der Maaten [2]. One of the key features of t-SNE is that it is able to handle nonlinear relationships between the data points, which makes it well-suited for visualizing complex data structures. It is also relatively robust to noise and can handle missing data.

The algorithm starts by transforming the high-dimensional Euclidean distances between data points into joint probabilities which contain information on similarities. In the case of data point $x_j$ for example, and data point $x_i$, the joint probability $p_ij$ defines the chances that $x_i$ woudl select $x_j$ as its neighbour if neighbours were chosen in relation to their probability density under a Gaussian distribution centered at $x_i$. This joint probability assumes elevated values for points that are located nearby, while it tends to zero in the opposite case. $p_ij$ is defined mathematically in Equation 1:

*Insert Equation 1 here.*

## Results

The results of the t-SNE analysis are presented graphically in Figure 1. The blue dots, which represent those slate sections with the lowest quality index (1) are easily clustered by the t-SNE. This implies a that those data points with a high quality are correlated. Continuing with the data points of a medium quality (2), these are more scattered but still present two important clusters. On the other hand, the green dots, which correspond to the lowest slate quality (3), are far more spread out than the other two categories. This is mainly due to the high number of variables that affect the quality index with different degrees of importance.

![**Figure 1**. Results of the t-SNE. Each point in the plot represents a slate section. The data points of the highest quality (3) are coloured in green. The two other quality classifications, medium (2) and low (1) are represented in red and blue, respectively.](Figures/figure1.png)

Nevertheless, the are several clusters that contain data points of more than one quality index. It is most noticeable the mostly-blue clump located between 0 and 10 in the horizontal axis and -5 and 0 in the vertical axis. This cluster contains data points of all three quality levels, which indicates that there are certain cases in which the t-SNE is not able to classify correctly. Additionally, the t-SNE is able to find some correlations within the data and the target variable, quality in this case, but does not provide any information regarding those variables that hold a higher weight in the prediction of the quality index.

## Bibliography

[1]
[2]