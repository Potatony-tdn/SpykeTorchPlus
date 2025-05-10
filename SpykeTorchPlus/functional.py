import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np


def pad(input, pad, value=0):
    r"""Applies 2D padding on the input tensor.

    Args:
        input (Tensor): The input tensor.
        pad (tuple): A tuple of 4 integers in the form of (padLeft, padRight, padTop, padBottom)
        value (int or float): The value of padding. Default: 0

    Returns:
        Tensor: Padded tensor.
    """
    return fn.pad(input, pad, value=value)

# pooling
def pooling(input, kernel_size, stride=None, padding=0):
    r"""Performs a 2D max-pooling over an input signal (spike-wave or potentials) composed of several input
    planes.

    Args:
        input (Tensor): The input tensor.
        kernel_size (int or tuple): Size of the pooling window.
        stride (int or tuple, optional): Stride of the pooling window. Default: None
        padding (int or tuple, optional): Size of the padding. Default: 0

    Returns:
        Tensor: The result of the max-pooling operation.
    """
    return fn.max_pool2d(input, kernel_size, stride, padding)

# unpooling
def unpooling(input, kernel_size, stride=None, padding=0, switches=None, output_size=None):
    r"""Performs a 2D unpooling over an input signal (spike-wave or potentials) composed of several input
    planes.

    Args:
        input (Tensor): The input tensor.
        kernel_size (int or tuple): Size of the unpooling window.
        stride (int or tuple, optional): Stride of the unpooling window. Default: None
        padding (int or tuple, optional): Size of the padding. Default: 0
        switches (Tensor, optional): The indices tensor from max_pool2d. Default: None

    Returns:
        Tensor: The result of the unpooling operation.
    """
    # Check if input is on MPS device
    device = input.device
    input = input.cpu()
    if switches is not None:
        switches = switches.cpu()
    result = fn.max_unpool2d(input, switches, kernel_size, stride, padding, output_size)
    return result.to(device)



def fire(currents, decay, threshold):
    """
    Compute spikes and potentials with decay and reset.

    Args:
        currents (Tensor): Tensor of input currents at each time step (T x ...).
        decay (float): Decay factor.
        threshold (float): Firing threshold.
        reset (float): Reset potential after firing (unused in this version).

    Returns:
        spikes (Tensor): Tensor of output spikes.
        potentials (Tensor): Tensor of membrane potentials.
    """
    T = currents.shape[0]
    potentials = torch.zeros_like(currents)
    spikes = torch.zeros_like(currents)
    V = torch.zeros_like(currents[0])  # Initial membrane potential
    
    for t in range(T):
        # Update potential with decay and current
        if t > 0:
            V = V * decay * (1.0 - spikes[t - 1]) + currents[t]
        else:
            V = V * decay + currents[t]
            
        # Store potential before spike generation
        potentials[t] = V.clone()
        
        # Generate spikes
        spikes[t] = (V >= threshold).float()

    return spikes, potentials

def fire_(potentials, threshold=None):
    r"""The inplace version of :func:`~fire`
    """
    if threshold is None:
        potentials[:-1] = 0
        potentials.sign_()
    else:
        cumulative_potential = torch.cumsum(potentials, dim=0)
        num_spikes = (cumulative_potential / threshold).floor()
        
        spikes = torch.where(
            cumulative_potential >= threshold,
            (cumulative_potential % threshold < potentials).float(),
            torch.zeros_like(potentials)
        )
        
        spike_sum = torch.cumsum(spikes, dim=0)
        potentials.copy_(cumulative_potential - (spike_sum * threshold))
        potentials.copy_(spikes)

def threshold(potentials, threshold=None):
    r"""Applies a threshold on potentials by which all of the values lower or equal to the threshold becomes zero.
    If :attr:`threshold` is :attr:`None`, only the potentials corresponding to the final time step will survive.

    Args:
        potentials (Tensor): The tensor of input potentials.
        threshold (float): The threshold value. Default: None

    Returns:
        Tensor: Thresholded potentials.
    """
    outputs = potentials.clone().detach()
    if threshold is None:
        outputs[:-1]=0
    else:
        fn.threshold_(outputs, threshold, 0)
    return outputs

def threshold_(potentials, threshold=None):
    r"""The inplace version of :func:`~threshold`
    """
    if threshold is None:
        potentials[:-1]=0
    else:
        fn.threshold_(potentials, threshold, 0)

# in each position, the most fitted feature will survive (first earliest spike then maximum potential)
# it is assumed that the threshold function is applied on the input potentials
def pointwise_inhibition(thresholded_potentials):
    """Performs point-wise inhibition between feature maps. For each spatial position:
    1. Finds the first neuron to spike across all features
    2. If multiple neurons spike at the same time, selects the one with highest potential
    3. Inhibits all other neurons at that position

    Args:
        thresholded_potentials (Tensor): Input potentials of shape [timesteps, features, height, width]

    Returns:
        Tensor: Inhibited potentials where only the winning neuron at each position retains its value
    """
    # Get maximum potential across features at each position and timestep
    max_potentials, feature_indices = torch.max(thresholded_potentials, dim=1, keepdim=True)
    
    # Convert to spike indicators (0 or 1)
    spike_indicators = (max_potentials > 0).float()
    
    # For each position, find the first timestep where any neuron spikes
    # Shape: [1, 1, height, width]
    first_spike_times = torch.argmax(spike_indicators, dim=0, keepdim=True)
    
    # Get the feature indices at the first spike times
    # Shape: [1, 1, height, width]
    winning_features = feature_indices.gather(0, first_spike_times)
    
    # Create inhibition mask
    # Shape: [1, features, height, width]
    inhibition_mask = torch.zeros_like(thresholded_potentials[0:1])
    
    # For each position, set the winning feature to 1, keeping all others at 0
    # arange creates indices for the feature dimension
    feature_indices = torch.arange(thresholded_potentials.size(1), device=thresholded_potentials.device)
    feature_indices = feature_indices.view(1, -1, 1, 1)
    inhibition_mask = (feature_indices == winning_features).float()
    # Apply inhibition mask to all timesteps
    return thresholded_potentials * inhibition_mask

# inhibiting particular features, preventing them to be winners
# inhibited_features is a list of features numbers to be inhibited
def feature_inhibition_(potentials, inhibited_features):
    r"""The inplace version of :func:`~feature_inhibition`
    """
    if len(inhibited_features) != 0:
        potentials[:, inhibited_features, :, :] = 0

def feature_inhibition(potentials, inhibited_features):
    r"""Inhibits specified features (reset the corresponding neurons' potentials to zero).

    Args:
        potentials (Tensor): The tensor of input potentials.
        inhibited_features (List): The list of features to be inhibited.

    Returns:
        Tensor: Inhibited potentials.
    """
    potentials_copy = potentials.clone().detach()
    if len(inhibited_features) != 0:
        feature_inhibition_(potentials_copy, inhibited_features)
    return potentials_copy
# returns list of winners
# inhibition_radius is to increase the chance of diversity among features (if needed)
def get_k_winners(potentials, kwta=1, inhibition_radius=0, spikes=None, input_spikes=None,verbose=False, similarity_threshold=0.8):
    """Finds k winners with proper pointwise and lateral inhibition, avoiding similar input patterns."""
    if spikes is None:
        spikes = potentials.sign()
    
    T, F, H, W = potentials.shape
    
    # Initialize first spike times
    spike_sums = spikes.sum(dim=0)
    spike_mask = spike_sums > 0
    first_spike_times = torch.full_like(spike_sums, T-1, dtype=torch.long)
    first_spike_times[spike_mask] = spikes[:, spike_mask].argmax(dim=0)

    winners = []
    current_time = -1
    spike_times = []
    
    # Store receptive fields of winners for similarity comparison
    winner_patterns = []
    
    while len(winners) < kwta:
        remaining_times = first_spike_times[first_spike_times > current_time]
        if len(remaining_times) == 0:
            break
            
        current_time = remaining_times.min()
        current_spike_mask = (first_spike_times == current_time) & spike_mask
        pot_values = potentials[current_time].clone()
        pot_values = pot_values * current_spike_mask.float()
        
        while len(winners) < kwta:
            max_pot, max_idx = pot_values.view(-1).max(0)
            if max_pot <= 0:
                break
                
            f, h, w = np.unravel_index(max_idx.item(), pot_values.shape)

            winners.append((current_time, f, h, w))
            spike_times.append(current_time)
            
            # Apply inhibition
            if inhibition_radius > 0:
                h_start = max(0, h - inhibition_radius)
                h_end = min(H, h + inhibition_radius + 1)
                w_start = max(0, w - inhibition_radius)
                w_end = min(W, w + inhibition_radius + 1)
                pot_values[:, h_start:h_end, w_start:w_end] = 0


    if verbose:
        if len(winners) == 0:
            print("No winners found!")
        else:
            print(f"Found {len(winners)} winners")
            print("Spike times:", sorted(spike_times))
            earliest = min(spike_times) if spike_times else None
            latest = max(spike_times) if spike_times else None
            print(f"Earliest spike: t={earliest}, Latest spike: t={latest}")
            
    return winners


# decrease lateral intencities by factors given in the inhibition_kernel
def intensity_lateral_inhibition(intencities, inhibition_kernel):
    r"""Applies lateral inhibition on intensities. For each location, this inhibition decreases the intensity of the
    surrounding cells that has lower intensities by a specific factor. This factor is relative to the distance of the
    neighbors and are put in the :attr:`inhibition_kernel`.

    Args:
        intencities (Tensor): The tensor of input intensities.
        inhibition_kernel (Tensor): The tensor of inhibition factors.

    Returns:
        Tensor: Inhibited intensities.
    """
    intencities.squeeze_(0)
    intencities.unsqueeze_(1)

    inh_win_size = inhibition_kernel.size(-1)
    rad = inh_win_size//2
    # repeat each value
    values = intencities.reshape(intencities.size(0),intencities.size(1),-1,1)
    values = values.repeat(1,1,1,inh_win_size)
    values = values.reshape(intencities.size(0),intencities.size(1),-1,intencities.size(-1)*inh_win_size)
    values = values.repeat(1,1,1,inh_win_size)
    values = values.reshape(intencities.size(0),intencities.size(1),-1,intencities.size(-1)*inh_win_size)
    # extend patches
    padded = fn.pad(intencities,(rad,rad,rad,rad))
    # column-wise
    patches = padded.unfold(-1,inh_win_size,1)
    patches = patches.reshape(patches.size(0),patches.size(1),patches.size(2),-1,patches.size(3)*patches.size(4))
    patches.squeeze_(-2)
    # row-wise
    patches = patches.unfold(-2,inh_win_size,1).transpose(-1,-2)
    patches = patches.reshape(patches.size(0),patches.size(1),1,-1,patches.size(-1))
    patches.squeeze_(-3)
    # compare each element by its neighbors
    coef = values - patches
    coef.clamp_(min=0).sign_() # "ones" are neighbors greater than center
    # convolution with full stride to get accumulative inhibiiton factor
    factors = fn.conv2d(coef, inhibition_kernel, stride=inh_win_size)
    result = intencities + intencities * factors

    intencities.squeeze_(1)
    intencities.unsqueeze_(0)
    result.squeeze_(1)
    result.unsqueeze_(0)
    return result

# performs local normalization
# on each region (of size radius*2 + 1) the mean value is computed and 
# intensities will be divided by the mean value
# x is a 4D tensor
def local_normalization(input, normalization_radius, eps=1e-12):
    r"""Applies local normalization. on each region (of size radius*2 + 1) the mean value is computed and the
    intensities will be divided by the mean value. The input is a 4D tensor.

    Args:
        input (Tensor): The input tensor of shape (timesteps, features, height, width).
        normalization_radius (int): The radius of normalization window.

    Returns:
        Tensor: Locally normalized tensor.
    """
    # computing local mean by 2d convolution
    kernel = torch.ones(1,1,normalization_radius*2+1,normalization_radius*2+1,device=input.device).float()/((normalization_radius*2+1)**2)
    # rearrange 4D tensor so input channels will be considered as minibatches
    y = input.squeeze(0) # removes minibatch dim which was 1
    y.unsqueeze_(1)  # adds a dimension after channels so previous channels are now minibatches
    means = fn.conv2d(y,kernel,padding=normalization_radius) + eps # computes means
    y = y/means # normalization
    # swap minibatch with channels
    y.squeeze_(1)
    y.unsqueeze_(0)
    return y

def pooling_with_switches(input, kernel_size, stride=None, padding=0):
    r"""Performs a 2D max-pooling over an input signal and returns both the pooled result
    and switch indices for unpooling.

    Args:
        input (Tensor): The input tensor.
        kernel_size (int or tuple): Size of the pooling window.
        stride (int or tuple, optional): Stride of the pooling window. Default: None
        padding (int or tuple, optional): Size of the padding. Default: 0

    Returns:
        tuple: (pooled_output, indices) where indices are the switch values for unpooling
    """
    return fn.max_pool2d(input, kernel_size, stride, padding, return_indices=True)
