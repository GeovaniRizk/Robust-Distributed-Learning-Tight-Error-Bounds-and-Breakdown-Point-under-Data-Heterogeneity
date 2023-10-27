import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision import datasets, transforms
import sys
import Dataset


def extreme_mnist(config):
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
    indices = dataset.targets == -1
    p = 0.1
    idx = torch.randperm(indices.size(0))[:int(indices.size(0)*p)]
    indices[idx] = True

    dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices]

    # targets = dataset.targets.numpy()
    dataset_dict = {}

    for i in range(config['nb_classes']):
        print(i)
        indices = dataset.targets == i
        idx = [j for j, x in enumerate(indices) if x]
        worker_dataset = torch.utils.data.Subset(dataset, idx)
        batch_size_train = len(idx)
        dataset_dict[i] = torch.utils.data.DataLoader(worker_dataset, batch_size=batch_size_train, shuffle=True)
    return dataset_dict


def compute_G(honest_gradients):
	nb_honest = len(honest_gradients)
	honest_gradient = torch.sum(torch.stack([gradient for gradient in honest_gradients]), dim = 0)/nb_honest
	G = torch.sum(torch.FloatTensor([torch.norm(gradient - honest_gradient)**2 for gradient in honest_gradients]))/nb_honest
	return G, honest_gradient


def compute_G_and_B_by_fixing_B(tab_G, tab_norm_honest):
	
	tab_G = np.array(tab_G)
	tab_norm_honest = np.array(tab_norm_honest)

	best_G = []
	best_B = []
	for f in range(1,10):
		n = 10+f
		tab_B = np.linspace(0, (n - 2 * f) / f, 100)[:-1]
		error_min = np.inf
		final_G = final_B = -1
		for B in tab_B:
			tmp = np.max([np.zeros(len(tab_G)), tab_G - B*tab_norm_honest], axis = 0)
			G = np.max(tmp)

			error = ((f/(n - 2*f)) * G) / (1- ((f/(n - 2*f)) * B))
			print(G,B, error)
			if error < error_min:
				error_min = error
				final_G = G
				final_B = B
		best_G.append(final_G)
		best_B.append(final_B)

	return np.array(best_G), np.array(best_B)









































