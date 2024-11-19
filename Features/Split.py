import numpy as np
import torch

def load_split(dataset, batch_size, train_split=0.7, val_split=0.2, random_seed=42):
  
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Split dataset into train, validation, and test sets
    train_end = int(np.floor(train_split * dataset_size))
    val_end = int(np.floor((train_split + val_split) * dataset_size))

    # Indices for each split
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Creating data samplers:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    # Creating data loaders:
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader