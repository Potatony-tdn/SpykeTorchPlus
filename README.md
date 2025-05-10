# SpykeTorch+

**SpykeTorch+** is a significantly modified fork of [SpykeTorch](https://github.com/miladmozafari/SpykeTorch), a high-speed simulator of convolutional spiking neural networks (SNNs) in which neurons emit at most one spike per stimulus.

This enhanced version addresses the limitations of the original code in handling realistic event-based datasets, such as those from event cameras (e.g., N-Caltech101), by introducing new dynamics, rewrites, and experimental techniques aimed at better capturing temporal patterns and optic flow.

<div align="center">
  <img src="https://raw.githubusercontent.com/miladmozafari/SpykeTorch/master/logo.png" alt="SpykeTorch logo" width="50%">
</div>

---

## What’s New in This Fork

### Major Improvements
- **Dynamic Decay Support**: Introduced decay factors in potential accumulation to handle moving patterns and improve temporal selectivity.
- **Flexible Firing Logic**: Rewrote `fire()` and `learn()` mechanisms to allow precise spiking, thresholding, and resetting of membrane potentials.
- **Temporal Learning Rate**: Learning rate now dynamically adapts based on spike timing proximity and receptive field contribution.
- **K-Winners Adaptation**: Switched from fixed KWTA to a more dynamic or even all-spike learning strategy after early training phases.
- **Pointwise Inhibition**: Added inhibition mechanisms across kernels and channels to prevent kernel collapse.
- **Improved STDP Rules**: Replaced static pre-post ordering with temporal relevance-based updates.

### Addressed Issues in Original Implementation
- Static input assumptions (spikes stay on once fired).
- Lack of decay support causing mushy activations on realistic data.
- Pre-post spike pairing based on fixed time windows rather than actual contribution.
- Kernel dominance due to output-channel-only inhibition.
- KWTA causing underutilization of late spikes.

### New Results
- Achieved **95.63% accuracy** on normalized N-Caltech101 data using:
  - Pointwise inhibition
  - Simple dynamic thresholding
  - Radicalized weight updates
- Preliminary visual inspection shows improved feature distinctiveness across layers, despite persistent dominance of horizontal patterns in early layers.

---

The original SpykeTorch was excellent for stylized, event-converted MNIST data. However, for real event data with high temporal resolution and motion, additional mechanisms are necessary to preserve the spatiotemporal richness and avoid overfitting to biased patterns.

This version aims to be a stepping stone toward more robust, biologically plausible, and efficient SNN training pipelines compatible with dynamic real-world data.

---

## 🚀 Getting Started

### Installation (via Conda)
```bash
git clone https://github.com/your-username/SpykeTorchPlus.git
cd SpykeTorchPlus
conda create -n spyketorchplus python=3.9
conda activate spyketorchplus
pip install -r requirements.txt
