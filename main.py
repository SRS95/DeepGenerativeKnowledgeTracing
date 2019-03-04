import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
import yaml
import argparse

from dataset import StudentInteractionsDataset
from model import RNN
from torch.utils.data import random_split, DataLoader
from train import train

SAMPLE_SEQ_LEN = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory of saving checkpoints')
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory of naive_c2_q50_s4000_v0.csv')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory of putting logs')
    parser.add_argument('--gpu', action='store_true', help="Turn on GPU mode")
    parser.add_argument('--train', action='store_true', default=False, help="Train model from scratch")
    parser.add_argument('--num_samples', type=int, default=4000, help="Number of samples to generate")

    args = parser.parse_args()
    return args



def dict2namespace(config):
    new_config = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(new_config, key, value)
    return new_config


def parse_config(args):
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return dict2namespace(config)


def main():
    args = parse_args()
    config = parse_config(args)

    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create character-level RNN for data of the form in dataset
    dataset = StudentInteractionsDataset(csv_file='data/naive_c5_q50_s4000_v1.csv', root_dir='data/')
    rnn = RNN(
        voc_len=dataset.voc_len,
        voc_freq = dataset.voc_freq,
        embedding_dim=config.embedding_dim,
        num_lstm_units=config.num_lstm_units,
        num_lstm_layers=config.num_lstm_layers,
        device=device
    )


    if args.train:
        # Split data into train and test sets
        train_size = int(0.8 * dataset.data.shape[0])
        test_size = dataset.data.shape[0] - train_size
        train_set, test_set = random_split(dataset, [train_size, test_size])

        # Create train and test dataloaders
        train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=config.batch_size, shuffle=True)

        train(args, rnn, train_loader, test_loader)

    else:
        rnn.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'model-07150.pt'), map_location=device))
        print("RNN weights restored.")

        samples = []
        for i in range(args.num_samples):
           samples.append(rnn.sample(SAMPLE_SEQ_LEN))

        samples = pd.DataFrame(samples)
        file_path = "data/generated/samples_" + str(args.num_samples) + ".csv"
        samples.to_csv(file_path, index=False)



if __name__ == '__main__':
    main()