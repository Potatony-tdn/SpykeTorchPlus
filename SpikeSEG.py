###################################################################################
# Enhanced Implementation of:                                                     #
# "STDP-based spiking deep convolutional neural networks for object recognition"  #
# Original paper: https://arxiv.org/abs/1611.01421                                #
#                                                                                 #
# Reference:                                                                      #
# Kheradpisheh, Saeed Reza, et al.                                                #
# "STDP-based spiking deep convolutional neural networks for object recognition." #
# Neural Networks 99 (2018): 56-67.                                               #
#                                                                                 #
# This enhanced version includes:                                                 #
# - Support for N-Caltech101 dataset                                              #
# - Extended STDP learning capabilities                                           #
# - Improved temporal dynamics                                                    #
# - Enhanced visualization and monitoring tools                                   #
#                                                                                 #
# Dataset Attribution:                                                            #
# The N-Caltech101 dataset used in this code is based on the original Caltech101  #
# dataset and was created by Garrick Orchard et al. The dataset is released under #
# the Creative Commons Attribution 4.0 license.                                   #
#                                                                                 #
# Reference:                                                                      #
# Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N. "Converting Static Image   #
# Datasets to Spiking Neuromorphic Datasets Using Saccades", Frontiers in         #
# Neuroscience, vol.9, no.437, Oct. 2015                                          #
#                                                                                 #
# Dataset available at: https://www.garrickorchard.com/datasets/n-caltech101      #
###################################################################################

from sys import exit

import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
import numpy as np
import SpykeTorchPlus.snn as snn
import SpykeTorchPlus.functional as sf
from torch.utils.data import Dataset
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import copy
import shutil
from torch.utils.data import random_split


# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    use_cuda = True  # We'll reuse this flag for MPS acceleration
    print("Using MPS (Metal) device")
else:
    device = torch.device("cpu")
    use_cuda = False
    print("MPS device not found, using CPU")

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = snn.Convolution(2, 4, 5, 0.8, 0.05)
        self.conv1_t = 10
        self.conv1_t_pass = 5
        self.k1 = 8
        self.r1 = 3
        self.conv1.weight = Parameter(torch.tensor([[[[0.25,0.25,0.25,0.25,0.25],
                                                     [0.25,0.25,0.25,0.25,0.25],
                                                     [0.75,0.75,0.75,0.75,0.75],
                                                     [0.25,0.25,0.25,0.25,0.25],
                                                     [0.25,0.25,0.25,0.25,0.25]]],
                                                   [[[0.75,0.75,0.25,0.25,0.25],
                                                     [0.75,0.75,0.25,0.25,0.25],
                                                     [0.75,0.75,0.25,0.25,0.25],
                                                     [0.75,0.75,0.25,0.25,0.25],
                                                   [0.75,0.75,0.25,0.25,0.25]]],
                                                   [[[0.75,0.25,0.25,0.25,0.25],
                                                     [0.25,0.75,0.75,0.25,0.25],
                                                     [0.25,0.75,0.75,0.75,0.25],
                                                     [0.25,0.25,0.75,0.75,0.25],
                                                     [0.25,0.25,0.25,0.25,0.75]]],
                                                   [[[0.25,0.25,0.25,0.25,0.75],
                                                     [0.25,0.25,0.75,0.75,0.25],
                                                     [0.25,0.75,0.75,0.75,0.25],
                                                     [0.25,0.75,0.75,0.25,0.25],
                                                     [0.75,0.25,0.25,0.25,0.25]]]]))
        self.conv1.weight = Parameter(self.conv1.weight.repeat(1,2,1,1))
        print("conv1 weight", self.conv1.weight.shape)

        self.conv2 = snn.Convolution(4, 8, 15, 0.8, 0.05)
        self.conv2_t = 75
        self.conv2_t_pass = 50
        self.k2 = 8
        self.r2 = 3

        self.conv3 = snn.Convolution(8, 2, 7, 0.8, 0.05)
        self.conv3_t = 40
        self.conv3_t_pass = 20
        self.k3 = 2
        self.r3 = 5

        self.stdp1 = snn.STDP(self.conv1, (0.015, -0.005), use_stabilizer=True)
        self.stdp2 = snn.STDP(self.conv2, (0.015, -0.005), use_stabilizer=True)
        self.stdp3 = snn.STDP(self.conv3, (0.015, -0.005), use_stabilizer=True)
        self.max_ap = Parameter(torch.Tensor([0.15]))

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0
        self.spk_cnt3 = 0
        self.convergence_history = {1: [], 2: [], 3: []}
        self.max_potentials = {1: [], 2: [], 3: []}  # Track max potentials for each layer

        self.feature_stats = {
            'mean': [],
            'std': [],
            'activation_ratio': []
        }

        self.vis_counter = 0  # Add this line
        self.vis_interval = 100000  # Show visualization every 100 samples

        # Add weight history tracking
        self.weight_history = {
            1: [],  # For conv1
            2: [],  # For conv2
            3: []   # For conv3
        }
        self.stdp_update_counter = 0
        self.rotation_interval = 100  # How often to check for feature rotation
        self.rotation_threshold = 0.8  # Correlation threshold for rotation
        self.iteration_count = 0
    
    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners

    def forward(self, input, max_layer):
        input = sf.pad(input.float(), (2,2,2,2), 0)
        spike_times = {}
        pool_switches = {}
        
        if self.training:
            pot = self.conv1(input)
            if max_layer != 1:  
                spk, pot = sf.fire(pot, 0.9, self.conv1_t_pass)
                self.weight_history[1].append(self.conv1.weight.data.clone().cpu().numpy())


            if max_layer == 1:
                spk, pot = sf.fire(pot, 0.9, self.conv1_t)
                winners = sf.get_k_winners(pot, self.k1, self.r1, spk, input_spikes=input, verbose=False)
                self.save_data(input, pot, spk, winners)
                return spk, pot



            spk_in = sf.pad(sf.pooling(spk, 2, 2, 1),(2,2,2,2),0)
            pot = self.conv2(spk_in)
            max_pot2 = pot.max().item()
            self.max_potentials[2].append(max_pot2)
            if max_layer != 2:  
                spk, pot = sf.fire(pot, 0.9, self.conv2_t_pass)
                self.weight_history[2].append(self.conv1.weight.data.clone().cpu().numpy())
            self.monitor_features(spk)

            if max_layer == 2:
                spk, pot = sf.fire(pot, 0.9, self.conv2_t)
                spk_count = spk.sum().item()
                if spk_count < 10000:
                    self.conv2_t = self.conv2_t-0.5
                if spk_count > 30000:
                    self.conv2_t = self.conv2_t+0.5
                winners = sf.get_k_winners(pot, self.k2, self.r2, spk, input_spikes=spk_in, verbose=False)
                self.save_data(spk_in, pot, spk, winners)
                  # Get the timestep with maximum activity
                winner_timestep = spk.sum(dim=(1,2,3)).argmax().cpu().item() + 1
                
                # Only visualize every vis_interval samples
                self.vis_counter += 1
                if self.vis_counter % self.vis_interval == 0:
                    input_cpu = input.cpu()
                self.weight_history[1].append(self.conv2.weight.data.clone().cpu().numpy())

                return spk, pot
        

            spk_in = sf.pad(sf.pooling(spk, 2, 2, 1),(2,2,2,2),0)
            pot = self.conv3(spk_in)
            max_pot3 = pot.max().item()
            self.max_potentials[3].append(max_pot3)
            spk, pot = sf.fire(pot,0.9, self.conv3_t)

            if max_layer != 3:
                spk, pot = sf.fire(pot,0.9, self.conv3_t_pass)
                self.weight_history[3].append(self.conv3.weight.data.clone().cpu().numpy())

            if max_layer == 3:
                spk, pot = sf.fire(pot,0.9, self.conv3_t)
                spk_count = spk.sum().item()
                if spk_count < 7000:
                    self.conv3_t = self.conv3_t-0.5
                if spk_count > 20000:
                    self.conv3_t = self.conv3_t+0.5
                winners = sf.get_k_winners(pot, self.k3, self.r3, spk, input_spikes=spk_in, verbose=False)
                self.save_data(spk_in, pot, spk, winners)
                self.weight_history[2].append(self.conv3.weight.data.clone().cpu().numpy())
                return spk, pot
            return spk
        else:
            # Evaluation mode - track all spike times and switches
            # Layer 1
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, 0.9, self.conv1_t_pass)
            spk, pool1_switches = sf.pooling_with_switches(spk, 2, 2, 1)
            spike_times['layer1'] = spk.clone()
            pool_switches['layer1'] = pool1_switches

            # Layer 2
            spk_in = sf.pad(spk, (2,2,2,2))
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, 0.9, self.conv2_t_pass)
            spk, pool2_switches = sf.pooling_with_switches(spk, 2, 2, 1)
            spike_times['layer2'] = spk.clone()
            pool_switches['layer2'] = pool2_switches

            # Layer 3
            spk_in = sf.pad(spk, (2,2,2,2))
            pot = self.conv3(spk_in)
            spk, pot = sf.fire(pot, 0.9, self.conv3_t_pass)
            spike_times['layer3'] = spk.clone()
            self.ctx['spike_times'] = spike_times
            self.ctx['pool_switches'] = pool_switches
            
            return spk, self.ctx

    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 3:
            self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def get_layer_convergence(self, layer_idx):
        if layer_idx == 1:
            weights = self.conv1.weight.data.cpu().numpy()
        elif layer_idx == 2:
            weights = self.conv2.weight.data.cpu().numpy()
        elif layer_idx == 3:
            weights = self.conv3.weight.data.cpu().numpy()
            
        conv = np.abs(weights * (1-weights)).sum() / np.prod(weights.shape)
        self.convergence_history[layer_idx].append(conv)
        return conv
    
    def monitor_features(self, layer_output):
        self.feature_stats['mean'].append(layer_output.mean().item())
        self.feature_stats['std'].append(layer_output.std().item())
        self.feature_stats['activation_ratio'].append((layer_output > 0).float().mean().item())

    def rotate_redundant_features(self, layer_idx):
        if layer_idx == 1:
            weights = self.conv1.weight
        elif layer_idx == 2:
            weights = self.conv2.weight
        else:
            weights = self.conv3.weight
            
        # Calculate feature similarities
        flat_weights = weights.view(weights.shape[0], -1)
        similarities = torch.mm(flat_weights, flat_weights.t())
        
        # Find highly correlated pairs
        corr_pairs = torch.where(similarities > self.rotation_threshold)
        
        # Rotate features to break symmetry
        for i, j in zip(*corr_pairs):
            if i >= j:  # Avoid duplicate pairs
                continue
            # Add random noise to break symmetry
            noise = torch.randn_like(weights[i]) * 0.1
            weights[i] = weights[i] + noise
            weights[j] = weights[j] - noise
            
        # Normalize weights after rotation
        weights.data = torch.clamp(weights.data, 0, 1)


def train_unsupervise(network, data, layer_idx):
    network.train()
    if use_cuda:
        data = data.to(device)
    data = data.squeeze(0)
    network(data, layer_idx)
    network.stdp(layer_idx)
    return network.ctx

def pass_through(network, data, layer_idx):
    network.eval()
    if use_cuda:
        data = data.to(device)
    data = data.squeeze(0)
    encoder_output, encoder_context = network(data, layer_idx)
    return encoder_output, encoder_context

def visualize_layer_output(encoder, data, layer_num):
    """
    Creates separate animated visualizations for each channel's spike outputs.
    
    Args:
        encoder: The encoder network
        data: Input data sample
        layer_num: The layer number being visualized
    """
    print("start")
    
    encoder.eval()  # Set to evaluation mode
    if use_cuda:
        data = data.to(device)
    data = data.squeeze(0)
    
    # Get layer output
    spikes, _ = encoder(data, layer_num)
    
    print("spikes:", spikes)


    # Move to CPU for visualization
    if spikes.device.type != 'cpu':
        spikes = spikes.cpu()
    
    timesteps, channels, height, width = spikes.shape
    
    # Create separate visualization for each channel
    for channel in range(channels):
        # Create figure for this channel
        fig = plt.figure(figsize=(15, 8))
        plt.suptitle(f'Layer {layer_num}, Channel {channel+1}/{channels} Output Spikes\n'
                     f'Shape: {spikes.shape} (Timesteps: {timesteps}, Height: {height}, Width: {width})')
        
        # Create heatmap
        img = plt.imshow(torch.zeros((height, width)), 
                        cmap='viridis',
                        animated=True,
                        vmin=0,
                        vmax=spikes[:, channel].max())
        plt.colorbar(img)
        
        # Add stats text box
        channel_spikes = spikes[:, channel]
        stats_text = f'Channel Statistics:\n'
        stats_text += f'Max spikes: {channel_spikes.max():.2f}\n'
        stats_text += f'Mean spikes: {channel_spikes.mean():.2f}\n'
        stats_text += f'Active pixels: {(channel_spikes > 0).float().mean():.1%}'
        
        plt.text(1.15, 0.95, stats_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        def update(frame):
            img.set_array(spikes[frame, channel])
            plt.title(f'Timestep {frame+1}/{timesteps}')
            return [img]
        
        anim = FuncAnimation(fig, 
                            update, 
                            frames=timesteps,
                            interval=200,  # 200ms between frames
                            blit=True)
        
        plt.show()


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.transconv1 = snn.TransposedConvolution(2, 8, 7, 0.8, 0.05)
        self.transconv1_t = 10

        self.transconv2 = snn.TransposedConvolution(8, 4, 15, 0.8, 0.05)
        self.transconv2_t = 5

        self.transconv3 = snn.TransposedConvolution(4, 2, 5, 0.8, 0.05)
        self.transconv3_t = 3

        # Weight initialization from encoder
        self.transconv1.weight.data = encoder.conv3.weight.data
        self.transconv2.weight.data = encoder.conv2.weight.data
        self.transconv3.weight.data = encoder.conv1.weight.data

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}

    def forward(self, input, encoder_ctx):
        spike_times = encoder_ctx['spike_times']
        pool_switches = encoder_ctx['pool_switches']

        # First transposed conv: 2 -> 48
        temporal_mask = spike_times['layer3']
        pot = input * temporal_mask
        pot = self.transconv1(pot)
        spk, pot = sf.fire(pot, 0.8, self.transconv1_t)

        temporal_mask = spike_times['layer2']
        spk = spk * temporal_mask

        spk = sf.unpooling(spk, 0, 2, switches=pool_switches['layer2'],output_size=(81,81))

        # Second transposed conv: 48 -> 8
        pot = self.transconv2(spk)
        spk, pot = sf.fire(pot, 0.8, self.transconv2_t)

        temporal_mask = spike_times['layer1']
        spk = spk * temporal_mask

        spk = sf.unpooling(spk, 0, 2, switches=pool_switches['layer1'],output_size=(180,180))

        # Third transposed conv: 8 -> 2
        pot = self.transconv3(spk)
        spk, pot = sf.fire(pot, 0.8, self.transconv3_t)

        return spk

class EventDataset(Dataset):
    def __init__(self, root_dir, classes=['Faces_easy', 'Motorbikes'], timesteps=120):
        self.root_dir = root_dir
        self.classes = classes
        self.samples = []
        self.timesteps = timesteps
        
        # Collect file paths and labels
        print("Collecting file paths...")
        
        # Get all samples for each class
        class_samples = []
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(root_dir, class_name)
            files = [f for f in os.listdir(class_path) if f.endswith('.bin')]
            class_samples.append([
                (os.path.join(class_path, file_name), class_idx)
                for file_name in files
            ])
        
        # Find minimum number of samples across classes
        min_samples = min(len(samples) for samples in class_samples)
        print(f"Balancing dataset to {min_samples} samples per class")
        
        # Randomly select equal number of samples from each class
        for class_samples_list in class_samples:
            # Randomly shuffle and take first min_samples
            random.shuffle(class_samples_list)
            self.samples.extend(class_samples_list[:min_samples])
        
        # Shuffle the combined balanced dataset
        random.shuffle(self.samples)
        
        # Print class distribution
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        print("Class distribution after balancing:")
        for class_idx, count in class_counts.items():
            print(f"Class {classes[class_idx]}: {count} samples")
        
        # Pre-load data with labels
        print("Pre-loading data into memory...")
        self.cached_data = []
        self.cached_labels = []
        for file_path, label in tqdm(self.samples, desc="Loading events"):
            input_events = self.read_events(file_path)
            self.cached_data.append(input_events)
            self.cached_labels.append(label)

    def read_events(self, file_path):
        # Read the binary file
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Convert to numpy array - each event is 5 bytes (40 bits)
        data_array = np.frombuffer(data, dtype=np.uint8).reshape(-1, 5)
        
        # Extract data according to bit format:
        # byte 0: X address (8 bits)
        # byte 1: Y address (8 bits)
        # bytes 2-4: polarity (1 bit) + timestamp (23 bits)
        x_coords = torch.from_numpy(data_array[:, 0])  # bits 39-32
        y_coords = torch.from_numpy(data_array[:, 1])  # bits 31-24
        
        # Get polarity from MSB of byte 2 (bit 23)
        polarities = (torch.from_numpy(data_array[:, 2]) & 0x80) >> 7
        
        # Get timestamp from remaining 23 bits (bits 22-0)
        # Combine: remaining 7 bits from byte 2, all 8 bits from byte 3, all 8 bits from byte 4
        timestamps = ((torch.from_numpy(data_array[:, 2]).long() & 0x7F) << 16) | \
                     (torch.from_numpy(data_array[:, 3]).long() << 8) | \
                      torch.from_numpy(data_array[:, 4]).long()
        
        # Define dimensions
        num_timesteps = 120  # We'll bin microsecond timestamps into 60 time steps
        height = 180
        width = 180
        num_channels = 2  # ON and OFF events
        
        # Normalize timestamps from microseconds to desired timesteps
        t_min = timestamps.min()
        t_max = timestamps.max()
        timesteps_normalized = ((timestamps - t_min).float() * (num_timesteps - 1) / (t_max - t_min)).long()
        
        # Create output tensor
        spike_tensor = torch.zeros((num_timesteps, num_channels, height, width))
        
        # Stack indices for efficient assignment
        indices = torch.stack([
            timesteps_normalized,  # Which time bin
            polarities,           # Which channel (ON/OFF)
            y_coords,            # Which row
            x_coords             # Which column
        ])
        
        # Filter valid events
        valid_events = (timesteps_normalized < num_timesteps) & \
                       (x_coords < width) & \
                       (y_coords < height)
        
        indices = indices[:, valid_events]
        
        # Place events in tensor
        spike_tensor.index_put_(
            (indices[0], indices[1], indices[2], indices[3]),
            torch.ones(indices.shape[1]),
            accumulate=False
        )
        # spike_tensor = sf.fire(spike_tensor, 0.8, 1)[0]
        
        # Normalize spike count while maintaining binary values
        target_spike_count = 50000  # Target number of spikes
        current_spikes = spike_tensor.sum()
        
        if current_spikes > target_spike_count:
            # Get all active locations
            active_locations = torch.nonzero(spike_tensor)
            
            # Create a new empty tensor
            new_tensor = torch.zeros_like(spike_tensor)
            
            # Randomly select target_spike_count indices
            selected_indices = torch.randperm(len(active_locations))[:target_spike_count]
            
            # Only keep the selected spikes
            keep_locations = active_locations[selected_indices]
            new_tensor[keep_locations[:, 0], keep_locations[:, 1], 
                      keep_locations[:, 2], keep_locations[:, 3]] = 1
            
            spike_tensor = new_tensor
        
        # Verify final spike count
        final_spikes = spike_tensor.sum()
        # print(f"Final spike count: {final_spikes} (target was {target_spike_count})")
        
        return spike_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return both data and label
        return self.cached_data[idx], self.cached_labels[idx]

# Create and analyze dataset
data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
full_dataset = EventDataset(data_root)

# Split dataset into train and test sets
# Use 80% for training, 20% for testing
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

encoder = Encoder()
if use_cuda:
    encoder = encoder.to(device)

decoder = Decoder()
if use_cuda:
    decoder = decoder.to(device)



def test(network, data):
    network.eval()
    if use_cuda:
        data = data.to(device)
    data = data.squeeze(0)
     
    # Get encoder output (using layer 3)
    output, _ = network(data, 3)
    
    # Sum activations across time and spatial dimensions for each channel
    channel_activations = output.sum(dim=[0,2,3])  # Shape will be (2,)
    print("channel_activations:", channel_activations)
    
    # Get the channel (0 or 1) with maximum activation
    predicted_label = torch.argmax(channel_activations).item()
    
    return predicted_label


def save_model(encoder, epoch, layer, metrics, save_dir='model_checkpoints'):
    """
    Save model state and training metrics
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(save_dir, f'encoder_layer{layer}_epoch{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'layer': layer,
        'model_state_dict': encoder.state_dict(),
        'convergence_history': encoder.convergence_history,
        'feature_stats': encoder.feature_stats,
        'metrics': metrics
    }, model_path)
    
    print(f"Model saved to {model_path}")

def load_model(encoder, path):
    """
    Load a saved model state
    """
    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.convergence_history = checkpoint['convergence_history']
    encoder.feature_stats = checkpoint['feature_stats']
    
    print(f"Model loaded from {path}")
    return checkpoint['metrics']

def train_layer(encoder, train_loader, layer_num, num_epochs=10, convergence_rate=0.01):
    """
    Train a single layer without intermediate saving
    """
    best_conv = float('inf')
    metrics = {
        'convergence_history': [],
        'final_convergence': None,
        'training_completed': False
    }
    
    for epoch in range(num_epochs):
        epoch_conv = []
        
        for data, _ in tqdm(train_loader, desc=f"Training layer {layer_num}"):
            train_unsupervise(encoder, data, layer_num)
            conv = encoder.get_layer_convergence(layer_num)
            epoch_conv.append(conv)
            
            if 0 <= conv < convergence_rate:
                print(f"\nLayer {layer_num} converged! Final convergence: {conv:.6f}")
                metrics['final_convergence'] = conv
                metrics['training_completed'] = True
                return metrics
        
        avg_conv = sum(epoch_conv) / len(epoch_conv)
        metrics['convergence_history'].append(avg_conv)
        print(f"Epoch {epoch} average convergence: {avg_conv:.6f}")
    
    return metrics

def run_training_trial(encoder, train_loader, trial_num, base_save_dir='model_trials'):
    """
    Run a complete training trial for all layers
    """
    trial_metrics = {}
    save_dir = os.path.join(base_save_dir, f'trial_{trial_num}')
    
    for layer in range(1, 4):  # 3 layers
        print(f"\nTraining Layer {layer} (Trial {trial_num})")
        metrics = train_layer(encoder, train_loader, layer, 
                            save_dir=save_dir, trial_num=trial_num)
        trial_metrics[f'layer_{layer}'] = metrics
        
    
    return trial_metrics

def evaluate_model(encoder, test_loader):
    """
    Evaluate model performance
    """
    predictions = []
    true_labels = []
    
    for data, _ in tqdm(test_loader, desc="Testing"):
        pred = test(encoder, data)
        predictions.append(pred)
        
        # Get true label
        idx = test_dataset.indices[len(true_labels)]
        file_path, _ = full_dataset.samples[idx]
        label = 0 if 'Faces_easy' in file_path else 1
        true_labels.append(label)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics
    accuracy = (predictions == true_labels).mean()
    conf_matrix = confusion_matrix(true_labels, predictions)
    class_report = classification_report(true_labels, predictions, 
                                      target_names=['Faces', 'Motorbikes'])
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'predictions': predictions,
        'true_labels': true_labels
    }

def save_trial_data(encoder, trial_num, accuracy, trial_metrics, base_save_dir=None):
    """
    Save all trial data including model weights and kernel visualizations
    """
    # Create timestamped directory for this run if it doesn't exist
    if base_save_dir is None:
        base_save_dir = 'model_checkpoints'
    
    # If run_dir is not provided, create a new timestamped directory
    if not os.path.exists(base_save_dir):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(base_save_dir, f'run_{timestamp}')
    else:
        run_dir = base_save_dir
        
    # Create trial directory
    trial_dir = os.path.join(run_dir, f'trial_{trial_num}')
    os.makedirs(trial_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(trial_dir, 'model.pth')
    torch.save({
        'trial': trial_num,
        'model_state_dict': encoder.state_dict(),
        'accuracy': accuracy,
        'metrics': trial_metrics
    }, model_path)
    
    # Save kernel visualizations for each layer
    plt.ioff()  # Turn off interactive mode
    for layer in range(1, 4):
        # Save basic kernel weights
        fig = plt.figure(figsize=(20, 20))
        weights_path = os.path.join(trial_dir, f'layer{layer}_kernels.png')
        plt.savefig(weights_path)
        plt.close(fig)
        
        # Save composite kernels
        fig = plt.figure(figsize=(20, 20))
        composite_path = os.path.join(trial_dir, f'layer{layer}_composite.png')
        plt.savefig(composite_path)
        plt.close(fig)
    
    print(f"Saved trial data to {trial_dir}")
    return run_dir

def main():
    config = {
        'num_trials': 5,
        'epochs_per_layer': 10,
        'base_save_dir': 'model_checkpoints'
    }
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config['base_save_dir'], f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    best_accuracy = 0.5
    best_trial = None
    best_encoder = None
    
    try:
        for trial in range(1, config['num_trials'] + 1):
            print(f"\nStarting Trial {trial}/{config['num_trials']}")
            
            # Initialize new encoder
            encoder = Encoder()
            if use_cuda:
                encoder = encoder.to(device)
            
            # Train all layers
            trial_metrics = {}
            for layer in range(1, 4):
                print(f"\nTraining Layer {layer}")
                layer_metrics = train_layer(encoder, train_loader, layer)
                trial_metrics[f'layer_{layer}'] = layer_metrics
            
            # Evaluate
            eval_metrics = evaluate_model(encoder, test_loader)
            accuracy = eval_metrics['accuracy']
            accuracy_bias = abs(accuracy - 0.5)
            
            print(f"\nTrial {trial} Results:")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Accuracy Bias: {accuracy_bias:.2%}")
            print(f"Confusion Matrix: \n{eval_metrics['confusion_matrix']}")
            print(f"Classification Report: \n{eval_metrics['classification_report']}")
            
            # Save trial data
            save_trial_data(encoder, trial, accuracy, trial_metrics, run_dir)
            
            # Update best model if needed
            if accuracy_bias > abs(best_accuracy - 0.5):
                best_accuracy = accuracy
                best_trial = trial
                best_encoder = copy.deepcopy(encoder)
                
                # Copy this trial's data to 'best_trial' folder
                best_dir = os.path.join(run_dir, 'best_trial')
                if os.path.exists(best_dir):
                    shutil.rmtree(best_dir)
                shutil.copytree(os.path.join(run_dir, f'trial_{trial}'), best_dir)
                print(f"New best model saved! Accuracy: {accuracy:.2%} (bias: {accuracy_bias:.2%})")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    
    finally:
        print(f"\nTraining Complete!")
        if best_trial is not None:
            print(f"Best Trial: {best_trial}")
            print(f"Best Accuracy: {best_accuracy:.2%}")
            print(f"Best Accuracy Bias: {abs(best_accuracy - 0.5):.2%}")
            print(f"Results saved in: {run_dir}")

if __name__ == "__main__":
    main()

