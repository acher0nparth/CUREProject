import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

'''unsupervised machine learning 
K means algorithm attempts to divide data points in to sectioned clusters of data based on classes
K value determines number of clusters 
first iteration randomly places centroids, establishes a line between them and cuts data in half
then center location of points is calculated through averaging and centroids are adjusted
process repeats until no changes to data points --> best possible cluster'''

digits = load_digits()
# .data represents features
# scaling saves time in computation
data = scale(digits.data)
y = digits.target

# len(np.unique(y)) calculates number of different classes
k = 10
samples, features = data.shape

# function within sklearn that automatically scores accuracy
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)
