import sklearn
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = load_digits()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model = KMeans(n_clusters=10, init="random", n_init=10)


# model.fit(x_train, y_train)
# model.score(x_test, y_test)
#
# predications = model.predict(x_test)
# true = 0
# false = 0
# for i in range(len(x_test)):
#     if predications[i] == y_test[i]:
#         true += 1
#     else:
#         false += 1
# print(true, false)
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean')))


bench_k_means(model, "1", X)
