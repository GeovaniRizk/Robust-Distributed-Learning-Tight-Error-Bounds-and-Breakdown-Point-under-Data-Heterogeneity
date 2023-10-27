import torch
import Dataset
import model
from torchvision import datasets, transforms
import sys

class Worker():
	def __init__(self, config, worker_id, dataloader, initial_weights, lbda = 0):
		'''
			id: int
			dataset: string
			initial_weights: list of tensors
		'''

		self.id = worker_id

		self.device = config['device']

		self.training_dataloader = dataloader

		self.nb_classes = config['nb_classes']

		self.model = model.logistic(self.nb_classes)

		if initial_weights is not None:
			self.set_model_parameters(initial_weights)

		self.model = self.model.to(self.device)

		self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

		self.lbda = lbda

		self.clip = None

	def get_id(self):
		return self.id

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
	def set_model_parameters(self, initial_weights):
		'''
			initial_weights: list of tensors
		'''
		for j, param in enumerate(self.model.parameters()):
			param.data = initial_weights[j].data.clone().detach()

	def set_model_gradient(self, flat_gradient):
		'''
			flat_gradient: flat tensor
		'''
		gradients = self.unflatten(flat_gradient, [param.grad for param in self.model.parameters()])

		for j, param in enumerate(self.model.parameters()):
			param.grad.data = gradients[j].data.clone().detach()

	def get_model_parameters(self):
		return [param for param in self.model.parameters()]
	def set_model_parameters_with_flat_tensor(self, flat_tensor):
		'''
			initial_weights: flat tensor
		'''
		list_of_parameters = self.unflatten(flat_tensor, self.get_model_parameters())
		for j, param in enumerate(self.model.parameters()):
			param.data = list_of_parameters[j].data.clone().detach()
		
	def get_flatten_model_parameters(self):
		return self.flatten([param for param in self.model.parameters()])

	def compute_gradient(self):
		self.model.train()
		self.model.zero_grad()

		inputs, targets = next(iter(self.training_dataloader))
		inputs, targets = inputs.to(self.device), targets.to(self.device)
		# print(len(inputs))
		outputs = self.model(inputs)

		loss = self.criterion(outputs, targets)

		loss.backward()

		grad = self.flatten([param.grad.data for param in self.model.parameters()]) + self.lbda * self.flatten([param.data for param in self.model.parameters()])

		return self.clip_gradient(grad), loss

	def clip_gradient(self, gradient):
		if self.clip is None:
			return gradient

		grad_norm = gradient.norm().item()
		if grad_norm > self.clip:
			gradient.mul_(self.clip / grad_norm)
		return gradient






