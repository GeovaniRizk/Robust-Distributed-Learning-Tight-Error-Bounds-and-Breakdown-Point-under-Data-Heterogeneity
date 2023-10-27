import torch
import Dataset
import model
from torchvision import datasets, transforms
import sys

class Server():
	def __init__(self, config):

		
		self.model = model.logistic(config['nb_classes'])

		self.model = self.model.to(config['device'])

		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['lr'])

		self.device = config['device']

		self.testing_dataset = datasets.MNIST(root='./data', train=False, download=True, transform = transforms.ToTensor())
		
		#Dataloaders
		self.testing_dataloader = torch.utils.data.DataLoader(self.testing_dataset, batch_size=len(self.testing_dataset), shuffle=False)

		self.model_size = len(torch.cat([param.view(-1) for param in self.model.parameters()]))

	def flatten(self, list_of_tensor):
		return torch.cat(tuple(tensor.view(-1) for tensor in list_of_tensor))

	def unflatten(self, flat_tensor, list_of_tensor):
		c = 0
		returned_list = [torch.zeros(tensor.shape) for tensor in list_of_tensor]
		for i, tensor in enumerate(list_of_tensor):
			count = torch.numel(tensor.data)
			returned_list[i].data = flat_tensor[c:c + count].view(returned_list[i].data.shape)
			c = c + count
		return returned_list

	def set_model_parameters(self, list_of_tensor):
		'''
			list_of_tensor: list of tensors
		'''
		for j, param in enumerate(self.model.parameters()):
			param.data = list_of_tensor[j].data.clone().detach()

	def set_model_parameters_with_flat_tensor(self, flat_tensor):
		'''
			initial_weights: flat tensor
		'''
		list_of_parameters = self.unflatten(flat_tensor, self.get_model_parameters())
		for j, param in enumerate(self.model.parameters()):
			param.data = list_of_parameters[j].data.clone().detach()

	def set_model_gradient_with_flat_tensor(self, flat_gradient):
		'''
			flat_gradient: flat tensor
		'''
		self.optimizer.zero_grad()

		gradients = self.unflatten(flat_gradient, [param for param in self.model.parameters()])
		for j, param in enumerate(self.model.parameters()):
			param.grad = gradients[j].clone().detach()
	
	def get_model_parameters(self):
		return [param.clone().detach() for param in self.model.parameters()]
	def get_flatten_model_parameters(self):
		return self.flatten([param.clone().detach() for param in self.model.parameters()])

	def step(self):

		self.optimizer.step()

	def set_model_parameters_to_zero(self):
		for param in self.model.parameters():
			param.data = torch.zeros(param.data.shape).to(self.device)

	def evaluate(self):
		with torch.no_grad():
			total = 0
			correct = 0 
			for data in self.testing_dataloader:
				inputs, targets = data
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.model(inputs)
				_, predicted = torch.max(outputs.data, 1)
				total += targets.size(0)
				correct += (predicted == targets).sum().item()
		return 100*(correct/total)

	def update_learning_rate(self, gamma):
		for g in self.optimizer.param_groups:
			g['lr'] = g['lr']*gamma
	def set_learning_rate(self, new_lr):
		for g in self.optimizer.param_groups:
			g['lr'] = new_lr



