import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from cobweb.cobweb_torch import CobwebTorchTree
from cobweb.visualize import visualize

# Configurations:
size_tr = 1000  # the size of the example training set
size_te = 10000  # the size of the example test set
normalize = False  # If true, normalize the MNIST training/test set
download = True  # Download a copy of MNIST dataset
seed = 123  # random seed used for shuffling
cuda = False  # We may just keep cuda=False since the training/predicting process will not be boosted with GPUs
verbose = True
random.seed(seed)



""" Load MNIST dataset and generate example training/test DataLoaders """

# Load the original MNIST dataset:
dataset_class = datasets.MNIST
transform = [transforms.ToTensor()]
if normalize:
	transform.append(transforms.Normalize((0.1307,), (0.3081,)))
dataset_transform = transforms.Compose(transform)
dataset_tr = dataset_class('./datasets/MNIST', train=True, download=download, transform=dataset_transform)
dataset_te = dataset_class('./datasets/MNIST', train=False, download=download, transform=dataset_transform)

# Return <DataLoader> object for the provided dataset object.
def get_data_loader(dataset, batch_size, cuda=cuda, drop_last=False, shuffle=False):
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
					  **({'num_workers': 0, 'pin_memory': True} if cuda else {}))


# Generate an example training DataLoader.
dataset_indices_tr = list(range(len(dataset_tr)))
random.shuffle(dataset_indices_tr)
dataset_indices_tr = dataset_indices_tr[:size_tr]
loader_tr = get_data_loader(Subset(dataset_tr, dataset_indices_tr), 
	batch_size=size_tr, cuda=cuda, drop_last=False, shuffle=True)


# Generate an example test DataLoader.
dataset_indices_te = list(range(len(dataset_te)))
random.shuffle(dataset_indices_te)
dataset_indices_te = dataset_indices_te[:size_te]
loader_te = get_data_loader(Subset(dataset_te, dataset_indices_te), 
	batch_size=size_te, cuda=cuda, drop_last=False, shuffle=True)



""" Initialize and Train Cobweb """
imgs_tr, labels_tr = next(iter(loader_tr))
tree = CobwebTorchTree(imgs_tr.shape[1:])
if verbose:
	print("Start Training.")
for i in tqdm(range(imgs_tr.shape[0])):
	tree.ifit(imgs_tr[i], labels_tr[i].item())
# Visualize Cobweb:
visualize(tree)



""" Make label predictions for the test set """
imgs_te, labels_te = next(iter(loader_te))
pred_labels = []
if verbose:
	print("Start Predicting.")
for i in tqdm(range(imgs_te.shape[0])):
	# Make a prediction:
	pred_probs = tree.predict_probs(imgs_te[i], None, max_nodes=50)
	pred_label = torch.tensor(sorted([(pred_probs[l], l) for l in pred_probs], reverse=True)[0][1])
	pred_labels.append(pred_label)

# Then you can return accuracy for the label predictions
correct = [1 if pred_labels[i] == labels_te[i] else 0 for i in range(len(pred_labels))]
accuracy = sum(correct) / len(imgs_te)
print(f"\nThe test accuracy of Cobweb/4V is {accuracy}.")














