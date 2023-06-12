import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the time series at the given index
        time_series = self.data[index]

        # Perform any necessary preprocessing on the time series
        # For example, you can normalize the data or apply any other transformations

        return time_series

def pad_time_series(time_series, length):
    # Pad or truncate the time series to the desired length
    padded_time_series = np.zeros((length, time_series.shape[1]))

    if len(time_series) >= length:
        padded_time_series[:length, :] = time_series[:length, :]
    else:
        padded_time_series[:len(time_series), :] = time_series

    return padded_time_series

def create_data_loader(time_series_data, batch_size, shuffle=True):
    # Create the dataset instance
    dataset = TimeSeriesDataset(time_series_data)

    # Get the maximum length among all time series
    max_length = max(len(ts) for ts in dataset)

    # Pad all time series to the maximum length
    padded_data = [pad_time_series(ts, max_length) for ts in time_series_data]

    # Create the data loader
    data_loader = DataLoader(
        padded_data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: torch.tensor(batch)
    )

    return data_loader

# Assuming you have a list of time series data
time_series_data = [...]

# Specify the batch size for the data loader
batch_size = 32

# Create the data loader
data_loader = create_data_loader(time_series_data, batch_size)

# Iterate over the data loader to get batches of equal-length time series
for batch in data_loader:
    # Process the batch as needed
    # For example, feed it to the first layer of your neural network
