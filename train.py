from model import RNN
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import utils as ut

num_epochs = 10000
learning_rate = 0.01


def test(rnn, test_loader):
	test_epoch_loss = 0.
	for batch_idx, data in enumerate(test_loader):
		test_epoch_loss += rnn.test_loss(data)
	return test_epoch_loss


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

		# Log summaries and save model every 50 epochs
		if epoch % 50 == 0: 
			ut.save_model(rnn, epoch, args.checkpoint_dir)
			test_loss = test(rnn, test_loader)
			ut.make_log(epoch_loss, test_loss, epoch, args.log_dir)


		print ("Train loss for epoch " + str(epoch) + ": " + str(epoch_loss))



