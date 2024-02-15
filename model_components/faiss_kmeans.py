# ---
# PANDAS
# Copyright (C) 2023 NAVER Corp.
# CC BY-NC-SA 4.0 license
# ---

import faiss
import numpy as np


class Faiss2SklearnKMeans:
    def __init__(self, d, k=100, nredo=10, niter=100, seed=42):
        self.k = k
        self.nredo = nredo
        self.niter = niter
        self.centroids = None
        self.kmeans = faiss.Kmeans(d=d,
                                   k=k,
                                   niter=self.niter,
                                   nredo=self.nredo,
                                   seed=seed)

    def fit(self, X):
        self.kmeans.train(X.astype(np.float32))
        self.centroids = self.kmeans.centroids

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]
