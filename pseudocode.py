"""
Pseudocode for creating a dataloader which can process multichannel, variable length time series.
"""
#open file (.pkl or similar) containing all time series data in the following shape:
#[ts1, ts2, ts3, ...] where 'tsk' is time series k which has the following format:
# [[ch1_1, ch1_2, ch1_3, ...],
#  [ch2_1, ch2_2, ch2_3, ...],
#  ...                       ] where 'chi_j' is measurement number j from the measurement channel i.

#specify batch size

#initialize data loader(time series data, batch size)
#   initialize custom dataset for time series data. This can include transformations or augmentations of the dataset.

#   for time series in dataset:
#       select random interval
#       do magic without further elaboration to get same length intervals
#   use the DataLoader module with batch size to initialize a data loader from the now generated time series
#   return the data loader

#use data:
#for batch in dataloader:
#   do machine learning


#open points:
#   -We should specify a minimum time series length before the for loop.
#   -Can a NN actually handle time series of random length? The loss function surely can, but is not what we're after.
