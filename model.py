import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, voc_len, voc_freq, embedding_dim, num_lstm_units, num_lstm_layers, device):
        super().__init__()
        
        # Number of distinct characterrs
        self.voc_len = voc_len
        
        # Prior over all characters
        self.voc_freq = torch.Tensor(voc_freq)

        # Trainable embeddings
        self.embedding = nn.Embedding(
            num_embeddings=voc_len,
            embedding_dim=embedding_dim
        )

        # Our model
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=num_lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # Make initial LSTM hidden state trainable
        # 1 in second dim because assuming we don't know batch size yet
        self.h_0 = nn.Parameter(torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size), requires_grad=True)
        self.c_0 = nn.Parameter(torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size), requires_grad=True)

        # Fully connected layer used to get logits
        self.h2o = nn.Linear(num_lstm_units, voc_len)

        self.device = device
        self.to(device)


    """
    Predict the next token's logits given an input token and a hidden state.
    :param input [torch.tensor]: The input token tensor with shape
        (batch_size, 1), where batch_size is the number of inputs to process
        in parallel.
    :param hidden [(torch.tensor, torch.tensor)]: The hidden state, or None if
        it's the first token.
    :return [(torch.tensor, (torch.tensor, torch.tensor))]: A tuple consisting of
        the logits for the next token, of shape (batch_size, num_tokens), and
        the next hidden state.
    """
    def forward(self, input, hidden):
        # If it's the first token
        if hidden == None:
            # Stretch h_0 and c_0 along second dim
            batch_size = input.shape[0]
            h_0 = self.h_0.repeat(1, batch_size, 1)
            c_0 = self.c_0.repeat(1, batch_size, 1)
            hidden = (h_0, c_0)

        # Convert characters to trainable embeddings and get lstm outputs
        embeddings = self.embedding(input)
        lstm, (h, c) = self.lstm(embeddings, hidden)

        # Pass lstm output through FC layer to get logits
        lstm = lstm.contiguous().view(-1, lstm.shape[2])
        logits = self.h2o(lstm)
        return logits, (h, c)


    # Compute loss over batch
    def loss(self, batch):
        # Batch size specified in config file
        # Assume all interactions are same length
        batch_size = batch.shape[0]
        seq_len = batch[0].shape[0]
 
        # Use prior over all characters to get the log-lik for first token
        nll = torch.sum(-torch.log(self.voc_freq[batch[:, 0]]))
        h_prev = None
        softmax = nn.Softmax(dim=1)

        for idx in range(seq_len - 1):
            curr_token = batch[:, idx].view(-1, 1)
            logits, h_prev = self.forward(curr_token, h_prev)
            log_probs = torch.log(softmax(logits))

            next_token = batch[:, idx + 1].view(-1, 1)
            next_token_onehot = torch.zeros(batch_size, self.voc_len)
            next_token_onehot.scatter_(1, next_token, 1)

            nll -= torch.sum(next_token_onehot * log_probs)
        
        return nll


    # Compute the negative log likelihood of a test batch of interactions
    def test_loss(self, batch):
        with torch.no_grad():
            return self.loss(batch)


    """
    Sample a student interaction string of length `seq_len` from the model.
    :param seq_len [int]: String length
    :return [list]: A list of length `seq_len` that contains each token in order.
                    Tokens should be numbers from {0, 1, 2, ..., voc_len}.
    """
    def sample(self, seq_len):
        voc_freq = self.voc_freq
        with torch.no_grad():
            # The starting hidden state of LSTM is None
            h_prev = None
            # Accumulate tokens into texts
            interaction = []
            # Randomly draw the starting token and convert it to a torch.tensor
            x = np.random.choice(voc_freq.shape[0], 1, p=voc_freq)[None, :]
            x = torch.from_numpy(x).type(torch.int64).to(self.device)
            
            # This will be used to compute probability distributions given logits
            softmax_calulator = torch.nn.Softmax()
            
            # Set curr_token to be the randomly chosen token
            curr_token = x

            while(len(interaction) < seq_len):
                # Append the current token to texts
                # Recall that we only want to append numbers, not tensors
                interaction.append(curr_token.numpy()[0][0])

                # Pass the current token and hidden state through the model
                next_logits, next_hidden_state = self.forward(curr_token, h_prev)

                # Use softmax to convert the logits into a probability distribution
                next_probabilities = np.reshape(softmax_calulator(next_logits).numpy(), (self.voc_len,))

                # Sample a token from the probability distribution computed by softmax
                # Set curr_token to be that token
                curr_token = np.random.choice(next_probabilities.shape[0], 1, p=next_probabilities)[None, :]
                curr_token = torch.from_numpy(curr_token).type(torch.int64).to(self.device)

                # Update curr token and hidden state accordingly
                h_prev = next_hidden_state

        return interaction


    """
    Compute the log-likelihood for each string in `strings`
    :param string [np.ndarray]: an integer array of length N.
    :return [float]: the log-likelihood
    """
    def compute_prob(self, string):
        voc_freq = self.voc_freq
        with torch.no_grad():
            # Initialize LSTM hidden state
            h_prev = None
            # Convert the starting token to a torch.tensor
            x = string[None, 0, None]
            x = torch.from_numpy(x).type(torch.int64).to(self.device)
            # The log-likelihood of the first token.
            # We will accumulate log-likelihoods of all other tokens to ll as well.
            ll = np.log(voc_freq[string[0]])

            # This will be used to compute probability distributions given logits
            softmax_calulator = torch.nn.Softmax()

            # This will be used in our while loop condition
            stringIndex = 1

            while(stringIndex < string.shape[0]):
                # Pass the current token and hidden state through the model
                next_logits, next_hidden_state = self.forward(x, h_prev)

                # Use softmax to convert the logits into a probability distribution
                next_probabilities = np.reshape(softmax_calulator(next_logits).numpy(), (self.voc_len,))

                # Add log likelihood into accumulator
                ll += np.log(next_probabilities[string[stringIndex]])

                # Prepare for next iteration
                h_prev = next_hidden_state
                x = string[None, stringIndex, None]
                x = torch.from_numpy(x).type(torch.int64).to(self.device)
                stringIndex += 1

            return ll

