from model import RNN
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

num_epochs = 1000
learning_rate = 0.001


def train(args, rnn, train_loader, test_loader):
	global num_epochs, learning_rate
	optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

	# Beginning of main training loop
	for epoch in tqdm(range(num_epochs)):
		
		# Loss over entire dataset for this epoch
		epoch_loss = 0.

		for batch_idx, data in enumerate(train_loader):
			# Get loss
			optimizer.zero_grad()
			loss = rnn.loss(data)
			epoch_loss += loss
			
			# Perform update
			loss.backward()
			optimizer.step()

		print ("Loss for epoch " + str(epoch) + ": " + str(epoch_loss))
			


			





	
