from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class ClusMetrics:
    def __init__(self, n_samples) -> None:
        self.score_funcs = [
            ("V-measure", metrics.v_measure_score),
            ("Rand index", metrics.rand_score),
            ("ARI", metrics.adjusted_rand_score),
            ("MI", metrics.mutual_info_score),
            ("NMI", metrics.normalized_mutual_info_score),
            ("AMI", metrics.adjusted_mutual_info_score),
        ]

        self.rng = np.random.RandomState(0)
        self.n_samples = 100
        self.n_clusters_range = np.linspace(2, self.n_samples, 10).astype(int)

    def random_labels(self, n_samples, n_classes):
        return self.rng.randint(low=0, high=n_classes, size=n_samples)

    def uniform_labelings_scores(self, score_func, n_samples, n_clusters_range, n_runs=5):
        scores = np.zeros((len(n_clusters_range), n_runs))

        for i, n_clusters in enumerate(self.n_clusters_range):
            for j in range(n_runs):
                labels_a = self.random_labels(n_samples = self.n_samples, n_classes = n_clusters)
                labels_b = self.random_labels(n_samples = self.n_samples, n_classes = n_clusters)
                scores[i, j] = score_func(labels_a, labels_b)
        return scores

    def report(self):
        print(self.n_clusters_range)
        plt.figure(2)

        plots = []
        names = []

        for marker, (score_name, score_func) in zip("d^vx.,", self.score_funcs):
            scores = self.uniform_labelings_scores(score_func, self.n_samples, self.n_clusters_range)
            plots.append(
                plt.errorbar(
                    self.n_clusters_range,
                    np.median(scores, axis=1),
                    scores.std(axis=1),
                    alpha=0.8,
                    linewidth=2,
                    marker=marker,
                )[0]
            )
            names.append(score_name)

        plt.title(
            "Clustering measures for 2 random uniform labelings\nwith equal number of clusters"
        )
        plt.xlabel(f"Number of clusters (Number of samples is fixed to {self.n_samples})")
        plt.ylabel("Score value")
        plt.legend(plots, names)
        plt.ylim(bottom=-0.05, top=1.05)
        plt.show()
        
dataset = pd.read_csv('../data/train_data.csv')
#screen_x left_iris_y
X = dataset.drop(columns = ['timestamp', 'left_iris_x', 'right_iris_x', 'right_iris_y', 'screen_y'])
print(X.head)

neighbors = NearestNeighbors(n_neighbors=20)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

istances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()
