import json
import random

import matplotlib.pyplot as plt
import torch
from cluster import cluster
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from cobweb.cobweb_torch import CobwebTorchTree
from cobweb.visualize import visualize

# Configurations:
seed = 100
random.seed(seed)
n_components = 2  # nbr of dimensions reduced to with PCA. otherwise you cannot visualize them in a plain figure!
n_split = 5  # the number of splits made. (not necessarily the number of clusters in the output partition)
assert n_split > 1, "the number of clusters should be greater than 1"
assert n_split <= 7, "the number of clusters should be less than 8 (in this example)"


""" Data loading and preprocessing """
with open('iris.json') as file:
	instances = json.load(file)
random.shuffle(instances)
# Remove the classifications they have
instance_no_class = [{k: instance[k] for k in instance if k != 'class'} for instance in instances]
classes = [instance[c] for instance in instances for c in instance if c == 'class']
# Transform the dictionaries into tensors
instance_tensors = [torch.tensor([instance[k] for k in instance]) for instance in instance_no_class]
# print(instance_tensors[0].shape)

""" Separate Train and Teset """
tree = CobwebTorchTree(instance_tensors[0].shape)
for i in tqdm(range(len(instance_tensors))):
    tree.ifit(instance_tensors[i], classes[i])
visualize(tree)

# pred_probs = tree.predict_probs(instance_tensors[15], None, max_nodes=50)
instance_te = instance_tensors[15]
pred_label = tree.categorize(instance_te).predict()[1]
print(pred_label)
# pred_label = sorted([(pred_probs[l], l) for l in pred_probs], reverse=True)[0][1]
print(pred_label, classes[15])


""" Train and make clustering """
tree = CobwebTorchTree(instance_tensors[0].shape)
cluster_iter = cluster(tree, instance_tensors, instance_tensors[0].shape, 
    min_split=1, max_split=n_split)  # An iterable, from n_split=min_splits to max_splits
if n_split > 1:
    for i in range(n_split - 1):
        clusters = next(cluster_iter)  # the one with n_split. You are free to visit the previous case.

# Compute the Rand index
ari = adjusted_rand_score(clusters, classes)

dv = DictVectorizer(sparse=False)
iris_X = dv.fit_transform(instance_no_class)
pca = PCA(n_components=n_components)
iris_2d_x = pca.fit_transform(iris_X)

colors = ['b', 'g', 'r', 'y', 'k', 'c', 'm']
shapes = ['o', '^', '+']
clust_set = {v: i for i, v in enumerate(list(set(clusters)))}
class_set = {v: i for i, v in enumerate(list(set(classes)))}

for class_idx, class_label in enumerate(class_set):
    x = [v[0] for i, v in enumerate(iris_2d_x) if classes[i] == class_label]
    y = [v[1] for i, v in enumerate(iris_2d_x) if classes[i] == class_label]
    c = [colors[clust_set[clusters[i]]] for i, v in enumerate(iris_2d_x) if
         classes[i] == class_label]
    plt.scatter(x, y, color=c, marker=shapes[class_idx], label=class_label)

plt.title("Cobweb Iris Clustering (ARI = %0.2f, n_split = %d)" % (ari, n_split))
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend(loc=4)
plt.show()



