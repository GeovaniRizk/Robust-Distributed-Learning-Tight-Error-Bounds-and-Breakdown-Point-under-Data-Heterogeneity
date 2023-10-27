import torch
import torchvision
import torchvision.transforms as transforms
import Dataset
import copy
import torch.multiprocessing as mp
import model
import numpy as np
import time
from worker import Worker
from server import Server
import random
import sys
from numpy import savetxt
import tools
import os
import json
import argparse

sys.path.append('../ByzLibrary')

from robust_aggregators import RobustAggregator
from byz_attacks import ByzantineAttack

random_seed = 1

torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--f', type=int, help='Number of Byzantine workers')
args = parser.parse_args()

config = {
	'T' : 10,
	'n' : 10,
	'f': 0,
	'lr' : 0.05,
	'milestones' : [350,420,480],
	'gamma' : 0.2,
	'dataset' : 'mnist',
	'nb_classes' : 10,
	'attack' : 'SF',
	'agg' : ['trmean', 'krum', 'median', 'geometric_median'],
	'optim' : 'gd',
	'lbdas' : 1e-4,
	'model' : 'logistic',
	'device' : device, 
	'nb_run' : 1, 
	'seed' : random_seed
}

config['nb_honest'] = 10

nb_trials = 10

save_folder = './save_fig2'

if not os.path.isdir(save_folder):
	os.mkdir(save_folder)

if not os.path.isdir(save_folder+'/data'):
	os.mkdir(save_folder+'/data')
if not os.path.isdir(save_folder+'/plots'):
	os.mkdir(save_folder+'/plots')

with open(save_folder+'/config.json', 'w') as fp:
	tmp = device
	config['device'] = str(tmp)
	json.dump(config, fp, indent=4)
	config['device'] = tmp

server = Server(config)
model_size = server.get_flatten_model_parameters()

print(save_folder+'data/Accuracies_'+str(0)+'_'+str(config['f'])+'_'+str(config['agg'][0])+'.csv')

tab_accuracies = np.zeros((config['nb_run'], len(config['agg']), nb_trials, config['T']))
tab_losses = np.zeros((config['nb_run'], len(config['agg']), nb_trials, config['T']))

for run in range(config['nb_run']):

	server = Server(config)
	testing_dataloader = server.testing_dataloader
	initial_server_weights = server.get_model_parameters()
	for idx_agg, agg in enumerate(config['agg']):
		for trial in range(nb_trials):
			config['n'] = 10 + trial
			config['f'] = trial
			regularization = config['lbdas']

			dataloaders = tools.extreme_mnist(config)
			
			attack = ByzantineAttack(config['attack'], config['f'], server.model_size, config['device'], 200, config['agg'])
			aggregator = RobustAggregator('nnm', agg, 1, config['f'], server.model_size, config['device'])
		
			print('n =', config['n'], 'f =', config['f'])

			server = Server(config)
			server.set_model_parameters(initial_server_weights)

			server.testing_dataloader = testing_dataloader

			honest_workers  = [Worker(config = config, worker_id = i, dataloader = dataloaders[i], initial_weights = initial_server_weights, lbda = regularization) for i in range(config['nb_honest'])]
			
			index_milestone = 0

			print(f'Epoch: -1 - Test accuracy: {server.evaluate():.2f}')
			for epoch in range(config['T']):
				params = []
				losses = []

				server_weights = server.get_model_parameters()

				for worker in honest_workers:
					worker.set_model_parameters(server_weights)

					grad, loss = worker.compute_gradient()
					params.append(grad)
					losses.append(loss.item())

				byz_params = attack.generate_byzantine_vectors(params, None, epoch)
				params = params + byz_params

				fed_avr_param = aggregator.aggregate(params)

				server.set_model_gradient_with_flat_tensor(fed_avr_param)
				server.step()

				if epoch == config['milestones'][index_milestone]:
					server.update_learning_rate(config['gamma'])
					if index_milestone < len(config['milestones'])-1:
						index_milestone = index_milestone + 1
					
				test_accuracy = server.evaluate()
				tab_accuracies[run][idx_agg][trial][epoch] = test_accuracy

				average_loss = np.mean(losses)
				tab_losses[run][idx_agg][trial][epoch] = average_loss


				if epoch % 1 ==0:
					print(f'Epoch: {epoch} - Test accuracy: {test_accuracy:.2f} - Average training loss: {average_loss:.5f}')

			savetxt(save_folder+'/data/Accuracies_'+str(run)+'_'+str(config['f'])+'_'+str(agg)+'.csv', tab_accuracies[run][idx_agg][trial], delimiter=',')
			savetxt(save_folder+'/data/Losses_'+str(run)+'_'+str(config['f'])+'_'+str(agg)+'.csv', tab_losses[run][idx_agg][trial], delimiter=',')









