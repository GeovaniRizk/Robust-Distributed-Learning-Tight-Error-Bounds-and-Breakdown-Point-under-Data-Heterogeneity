from numpy import genfromtxt
import numpy as np
from numpy import linalg as LA
import sys
import torch
import matplotlib.pyplot as plt
from numpy import savetxt
import os


sys.path.append('../ByzLibrary')

from robust_aggregators import RobustAggregator
from byz_attacks import ByzantineAttack



def loss_function(theta, X, y, lbda):
	return LA.norm(X@theta - np.reshape(y, (-1,1)))**2/(2*len(X)) + lbda * LA.norm(theta)**2/2

def grad_function(theta, X, y, lbda):
	return np.reshape((X.T @ X /len(X) + config['lbda'] * np.eye(d)) @  theta, (-1,1)) - (X.T @ np.reshape(y, (-1,1)))/len(X)

random_seed = 1
np.random.seed(random_seed)
torch.manual_seed(random_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
	'T' : 40000,
	'n' : 20,
	'f': np.arange(1,20),
	'lr' : 0.0001,
	'attack' : ['SF', 'FOE', 'ALIE', 'mimic'],
	'agg' : 'trmean',
	'lbda' : 1/np.sqrt(7),
	'seed' : random_seed,
	'device' : device, 
}

save_folder = 'gap_new_attack'

save_folder = './save_fig4'

if not os.path.isdir(save_folder):
	os.mkdir(save_folder)


data = genfromtxt("./mg_scaled.csv", delimiter=',')

nb_honest = 7
data = np.random.permutation(data)

p = 0.8

training_dataset = data[:int(len(data)*p):]

test_dataset  = data[int(len(data)*p):]

one_train = np.ones((len(training_dataset), 1))
one_test = np.ones((len(test_dataset), 1))

train_y, train_x, test_y, test_x = training_dataset[:,0], np.concatenate((one_train, training_dataset[:,1:]), axis=1) , test_dataset[:,0], np.concatenate((one_test, test_dataset[:,1:]), axis=1)

d = len(train_x[0])
n = config['n']
train_y, train_x, test_y, test_x = train_y[:nb_honest], train_x[:nb_honest], test_y[:nb_honest], test_x[:nb_honest]

eig_values, eig_vector = LA.eig( (train_x.T @ train_x ) / len(train_x) + config['lbda'] * np.eye(d))
L, mu = eig_values[0], eig_values[-1]

print(L,mu)

savetxt(save_folder+'/L_mu.csv', [L,mu], delimiter=',')

worker_datasets = [(train_x[int(i*(len(train_x)/nb_honest)):int((i+1)*(len(train_x)/nb_honest))], train_y[int(i*(len(train_x)/nb_honest)):int((i+1)*(len(train_x)/nb_honest))]) for i in range(nb_honest)]

server_parameter_0 = np.zeros((len(train_x[0]),1))

losses = [[[] for j in range(len(config['attack']))] for i in range(len(config['f']))]

theta_star = LA.inv((train_x.T @ train_x) + len(train_x)*config['lbda']*np.eye(d)) @ train_x.T @ np.reshape(train_y, (-1,1))
loss_star = loss_function(theta_star, train_x, train_y, config['lbda'])

savetxt(save_folder+'/theta_star.csv', theta_star, delimiter=',')
savetxt(save_folder+'/loss_star.csv', [loss_star], delimiter=',')

sum_ = 0
for worker in range(nb_honest):
	X, y = worker_datasets[worker]

	grad = grad_function(theta_star, X, y, config['lbda'])
	sum_ = sum_ + LA.norm(grad)**2

sum_ = sum_/nb_honest

savetxt(save_folder+'/sum_norm_theta_star.csv', [sum_], delimiter=',')

tab_kappa = []
for a, f in enumerate(config['f']):
	max_kappa = -1
	for b, attack_name in enumerate(config['attack']):

		server_parameter = np.copy(server_parameter_0)

		attack = ByzantineAttack(attack_name, f, d, config['device'], 200, config['agg'])
		aggregator = RobustAggregator('nnm', config['agg'], 1, f, d, config['device'])

		lr = config['lr']
		last_loss = np.inf
		loss = loss_function(server_parameter, train_x, train_y, config['lbda'])
		losses[a][b].append(loss)
		for t in range(config['T']):
			params = []
			for worker in range(nb_honest):
				worker_parameter = np.copy(server_parameter)

				X, y = worker_datasets[worker]
				grad = grad_function(worker_parameter, X, y, config['lbda'])
				params.append(torch.FloatTensor(grad))

			byz_params = attack.generate_byzantine_vectors(params, None, t)

			params = params + byz_params

			aggregated_gradient = aggregator.aggregate(params).numpy()

			avg_honest = torch.mean(torch.stack(params[:nb_honest]), dim=0)

			norm_diff_agg_avg = torch.norm(torch.FloatTensor(aggregated_gradient) - avg_honest)**2

			sum_norm_diff_honest_avg_honest = 0 
			for i in range(nb_honest):
				sum_norm_diff_honest_avg_honest = sum_norm_diff_honest_avg_honest + torch.norm(params[i] - avg_honest)**2
			sum_norm_diff_honest_avg_honest = sum_norm_diff_honest_avg_honest/nb_honest

			kappa = torch.sqrt(norm_diff_agg_avg/sum_norm_diff_honest_avg_honest)

			if max_kappa < kappa:
				max_kappa = kappa

			server_parameter = server_parameter - lr * aggregated_gradient
			loss = loss_function(server_parameter, train_x, train_y, config['lbda'])

			losses[a][b].append(loss)
			if t % 1000 == 0:
				print('t = ', t, ' - ', loss)
		savetxt(save_folder+'/Losses_'+str(f)+'_'+config['attack'][b]+'.csv', losses[a][b], delimiter=',')
		if f == 0:
			break

	print('kappa = ', max_kappa)

	tab_kappa.append(max_kappa)

savetxt(save_folder+'/Kappas_.csv', tab_kappa, delimiter=',')








