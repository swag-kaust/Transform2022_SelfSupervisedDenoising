import numpy as np
import random
import itertools
from scipy.signal import filtfilt, butter
import matplotlib.pyplot as plt
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


def add_bandlimited_noise(d, lc=2, hc=80, sc=0.5):
    n = band_limited_noise(size=d.shape, lowcut=lc, highcut=hc)

    return d + (n * sc), n


def add_trace_wise_noise(d,
                         num_noisy_traces,
                         noisy_trace_value,
                         num_realisations,
                        ):  
    
    alldata=[]
    for k in range(len(d)):        
        clean=d[k]    
        data=np.ones([num_realisations,d.shape[1],d.shape[2]])
        for i in range(len(data)):    
            corr = np.random.randint(0,d.shape[2], num_noisy_traces) 
            data[i] = clean.copy()
            data[i,:,corr] = np.ones([1,d.shape[1]])*noisy_trace_value
        alldata.append(data)
        
    alldata=np.array(alldata) 
    alldata=alldata.reshape(num_realisations*d.shape[0],d.shape[1],d.shape[2])
    print(alldata.shape)

    return alldata


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def array_bp(data, lowcut, highcut, fs, order=5):
    bp = np.vstack([butter_bandpass_filter(data[:, ix], lowcut, highcut, fs, order)
                    for ix in range(data.shape[1])])

    return bp


def band_limited_noise(size, lowcut, highcut, fs=250):

    basenoise = np.random.normal(size=size)
    # Pad top and bottom due to filter effects
    basenoise_pad = np.vstack([np.zeros([50, size[1]]), basenoise, np.zeros([50, size[1]])])
    # Bandpass base noise
    bpnoise =  array_bp(basenoise_pad, lowcut, highcut, fs, order=5)[:,50:-50]

    return bpnoise.T

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


def plot_corruption(noisy, 
                    crpt, 
                    mask, 
                    seismic_cmap='RdBu', 
                    vmin=-0.25, 
                    vmax=0.25):
    
    fig,axs = plt.subplots(1,3,figsize=[15,5])
    axs[0].imshow(noisy, cmap=seismic_cmap, vmin=vmin, vmax=vmax)
    axs[1].imshow(crpt, cmap=seismic_cmap, vmin=vmin, vmax=vmax)
    axs[2].imshow(mask, cmap='binary_r')

    axs[0].set_title('Original')
    axs[1].set_title('Corrupted')
    axs[2].set_title('Corruption Mask')
    
    fig.tight_layout()
    return fig,axs

def plot_training_metrics(train_accuracy_history,
                         test_accuracy_history,
                          train_loss_history,
                          test_loss_history
                         ):
    fig,axs = plt.subplots(1,2,figsize=(15,4))
    
    axs[0].plot(train_accuracy_history, 'r', lw=2, label='train')
    axs[0].plot(test_accuracy_history, 'k', lw=2, label='validation')
    axs[0].set_title('RMSE', size=16)
    axs[0].set_ylabel('RMSE', size=12)

    axs[1].plot(train_loss_history, 'r', lw=2, label='train')
    axs[1].plot(test_loss_history, 'k', lw=2, label='validation')
    axs[1].set_title('Loss', size=16)
    axs[1].set_ylabel('Loss', size=12)
    
    for ax in axs:
        ax.legend()
        ax.set_xlabel('# Epochs', size=12)
    fig.tight_layout()
    return fig,axs


def plot_synth_results(clean, 
                       noisy, 
                       denoised,
                       cmap='RdBu', 
                       vmin=-0.25, 
                       vmax=0.25):
    
    fig,axs = plt.subplots(1,4,figsize=[15,4])
    axs[0].imshow(clean, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].imshow(noisy, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[2].imshow(denoised, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[3].imshow(noisy-denoised, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    axs[0].set_title('Clean')
    axs[1].set_title('Noisy')
    axs[2].set_title('Denoised')
    axs[3].set_title('Noise Removed')

    fig.tight_layout()
    return fig,axs