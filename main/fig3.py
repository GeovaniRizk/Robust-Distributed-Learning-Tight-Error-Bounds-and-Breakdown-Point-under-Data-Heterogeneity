import torch
import torchvision
import numpy as np
from worker import Worker
from server import Server
import random
from numpy import savetxt
import tools
import os
import json
import sys

sys.path.append('../ByzLibrary')

from robust_aggregators import RobustAggregator
from byz_attacks import ByzantineAttack

random_seed = 1

torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = {
	'T' : 500,
	'dataset' : 'mnist',
	'nb_classes' : 10,
	'model' : 'logistic',
	'nb_honest' : 10,
	'device' : device, 
	'seed' : random_seed,
	'lr': 0
}

save_folder = './save_fig3'

if not os.path.isdir(save_folder):
	os.mkdir(save_folder)

server = Server(config)
model_size = server.get_flatten_model_parameters()

nb_samples = 100

tab_var_gradients = np.zeros(nb_samples+1)
tab_norm_honest = np.zeros(nb_samples+1)

dataloaders = tools.extreme_mnist(config)

server = Server(config)

theta_star = torch.load('tensor.pt')

honest_workers  = [Worker(config = config, worker_id = i, dataloader = dataloaders[i], initial_weights = None, lbda = 1e-4) for i in range(config['nb_honest'])]
		
theta_bis = torch.ones(server.model_size)*10000


for i in range(nb_samples+1):
	theta = theta_star * (1 - i/nb_samples) + theta_bis * (i/nb_samples)

	params = []
	losses = []

	for worker in honest_workers:
		worker.set_model_parameters_with_flat_tensor(theta)
		
		grad, loss = worker.compute_gradient()
		params.append(grad)
		losses.append(loss.item())


	var, honest_gradient = tools.compute_G(params)

	tab_norm_honest[i] = torch.norm(honest_gradient)**2
	tab_var_gradients[i] = var


G_us, B, = tools.compute_G_and_B_by_fixing_B(tab_var_gradients, tab_norm_honest)

f = np.arange(1,10)
n = f + 10

tab_error = ((f/(n - 2*f)) * G_us) / (1- ((f/(n - 2*f)) * B))
G_max = np.max(tab_var_gradients)

savetxt(save_folder+'/error_ours.csv', tab_error, delimiter=',')
savetxt(save_folder+'/error_old.csv', (f/(n - 2*f)) * G_max, delimiter=',')























