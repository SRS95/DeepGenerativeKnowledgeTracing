import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class StudentInteractionsDataset(Dataset):
    

    def __init__(self, csv_file, root_dir, transform=None):
        # Required by Pytorch dataset class
        self.data = pd.read_csv(csv_file, header=None).values
        self.root_dir = root_dir
        self.transform = transform

        # I'm adding these
        self.vocabulary = self.get_vocabulary()
        self.voc_len = len(self.vocabulary)
        self.char2idx = {c: i for i, c in enumerate(self.vocabulary)}
        self.voc_freq = self.get_voc_freq()

    
    # Distinct characters used
    def get_vocabulary(self):
        vocabulary = set()
        
        for row in self.data:
            for character in row:
                vocabulary.add(character)

        return sorted(list(vocabulary))


    # Distribution over words in the vocabulary
    def get_voc_freq(self):
        voc_freq = np.zeros(len(self.vocabulary))
        
        for row in self.data:
            for character in row:
                voc_freq[self.char2idx[character]] += 1
        
        voc_freq /= np.sum(voc_freq)
        return voc_freq


    # Override the __len__ method
    def __len__(self):
        return len(self.data)


    # Override the __getitem__ method so that dataset[i]
    # can be used to get the i-th sample
    def __getitem__(self, idx):
        return self.data[idx]



