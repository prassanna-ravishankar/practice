"""Experiment with soft-thresholded k-means feature for MNIST classification

This is experiment is a tentative alternative to approximate kernel expansions
explored on the same dataset by @amueller on this blog post:

http://peekaboo-vision.blogspot.fr/2012/12/kernel-approximations-for-efficient.html

Meant to be run with ``%run script.py`` in IPython.

The 1000-dim k-means based feature expansion should yield ~96% test accuracy
when trained on 20k samples in less than 20s (unsupervised feature extraction
+ classifier training).

The baseline linear model is accuracy 91% on the same dataset.

"""

# Author: olivier.grisel@ensta.org
# License: Simplified BSD
from time import time
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC

mnist = fetch_mldata('MNIST original')

# Load 30k samples in the dev set as we will use 3-folds CV, hence 20k samples
# for each training set.
X_dev, X_test, y_dev, y_test = train_test_split(
    mnist.data.astype(np.float32), mnist.target, train_size=30000,
    random_state=1)

scaler = MinMaxScaler()
X_scaled_dev = scaler.fit_transform(X_dev)
X_scaled_test = scaler.transform(X_test)

print("n_samples=%d, n_features=%d" % X_dev.shape)
print("n_classes=%d" % np.unique(y_dev).shape[0])


class MiniBatchKMeansMapper(MiniBatchKMeans):
    """Soft thresholding cosine transfomer k-means

    This is some kind poors man, non linear sparse coded feature mapping.

    """

    def _transform(self, X):
        # Compute cosine similarities of samples w.r.t. k-means centers
        # TODO: optim normalize the centers ones and for all
        c = normalize(self.cluster_centers_)
        X = normalize(X)
        sims = np.dot(X, c.T)

        # Remove the negative cosine features (~%50% of them)
        # TODO: make it possible to use a percentile or an absolute parameter
        # in range (-1, 1) to be cross-validated
        sims[sims < 0.0] = 0.0

        # Project the new features on the unit euclidean ball because it
        # seems reasonable...
        # TODO: make normalization optional to be cross validated
        return normalize(sims, copy=True)


mapper = MiniBatchKMeansMapper(
            n_clusters=1000, n_init=1, init='random', batch_size=1000,
            init_size=3000, random_state=1, verbose=0,
            compute_labels=False)

models = [
    LinearSVC(C=0.01, random_state=1),
    Pipeline([
        # Reduce dimensionality to make K-Means converge faster
        ('dim_reduction', RandomizedPCA(50, whiten=True, random_state=1)),

        # Non linear feature extraction akin to an approximate kernel
        # expansion
        ('feature_map', mapper),

        # Linear classification
        ('svm', LinearSVC(C=1, random_state=1)),
    ]),
]

def bench(model, X, y, cv=3):
    print("Computing %d-CV for %r..." % (cv, model))
    t0 = time()
    scores = cross_val_score(model, X, y, cv=cv, verbose=1, n_jobs=1)
    time_linear = time() - t0
    print("score: %0.3f +/- %0.3f" % (np.mean(scores), np.std(scores)))

    # compute duration for 1 fold, assuming n_jobs=1
    duration = time_linear / scores.shape[0]
    print("duration: %0.3fs" % duration)
    return np.mean(scores), duration

results = [bench(m, X_scaled_dev, y_dev) for m in models]
