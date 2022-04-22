import numpy as np
import random
import itertools
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader 


def regular_patching_2D(data, patchsize=[64, 64], step=[16, 16], verbose=True):
    """ Regular sample and extract patches from a 2D array
    :param data: np.array [y,x]
    :param patchsize: tuple [y,x]
    :param step: tuple [y,x]
    :param verbose: boolean
    :return: np.array [patch#, y, x]
    """

    # find starting indices
    x_start_indices = np.arange(0, data.shape[0] - patchsize[0], step=step[0])
    y_start_indices = np.arange(0, data.shape[1] - patchsize[1], step=step[1])
    starting_indices = list(itertools.product(x_start_indices, y_start_indices))

    if verbose:
        print('Extracting %i patches' % len(starting_indices))

    patches = np.zeros([len(starting_indices), patchsize[0], patchsize[1]])

    for i, pi in enumerate(starting_indices):
        patches[i] = data[pi[0]:pi[0]+patchsize[0], pi[1]:pi[1]+patchsize[1]]

    return patches


def add_whitegaussian_noise(d, sc=0.5):
    n = np.random.normal(size=d.shape)

    return d + (n * sc), n


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight)
        nn.init.constant(m.bias, 0)
        

def make_data_loader(noisy_patches, 
                     corrupted_patches, 
                     masks, 
                     n_training,
                     n_test,
                     batch_size,
                     torch_generator
                    ):
    
    # Define Train Set
    # Remember to add 1 to 2nd dim - Pytorch is [#data, #channels, height, width]
    train_X = np.expand_dims(corrupted_patches[:n_training],axis=1)
    train_y = np.expand_dims(noisy_patches[:n_training],axis=1)    
    msk = np.expand_dims(masks[:n_training],axis=1)   
    # Convert to torch tensors and make TensorDataset
    train_dataset = TensorDataset(torch.from_numpy(train_X).float(), 
                                  torch.from_numpy(train_y).float(), 
                                  torch.from_numpy(msk).float(),)

    # Define Test Set
    test_X = np.expand_dims(corrupted_patches[n_training:n_training+n_test],axis=1)
    test_y = np.expand_dims(noisy_patches[n_training:n_training+n_test],axis=1)
    msk = np.expand_dims(masks[n_training:n_training+n_test],axis=1) 
    test_dataset = TensorDataset(torch.from_numpy(test_X).float(), 
                                 torch.from_numpy(test_y).float(), 
                                 torch.from_numpy(msk).float(),)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch_generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader