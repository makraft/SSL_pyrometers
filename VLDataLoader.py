import numpy

samples=numpy.random.choice(10, size=(3, 8))
print(samples)

import torch
from torch.utils.data import DataLoader

class VariableLengthDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    
    def collate_fn(self, batch):
        # Customize the collate_fn method to handle variable-length sequences
        # batch is a list of (sequence, label) pairs, where sequence is a variable-length data series
        
        # Sort the batch in descending order of sequence lengths
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        
        # Extract the sequences and labels
        sequences, labels = zip(*batch)
        
        # Pad the sequences to the length of the longest sequence in the batch
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        
        return sequences_padded, torch.tensor(labels)

# Usage example
# Assuming you have a custom dataset called CustomDataset

dataset = CustomDataset()  # Initialize your custom dataset
custom_dataloader = VariableLengthDataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
