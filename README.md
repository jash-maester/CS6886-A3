# MobileNetV2 Quantization on CIFAR-10

This repository implements training and post-training quantization (PTQ) for MobileNetV2 on the CIFAR-10 dataset. The quantization method uses configurable bit-widths for both weights and activations to achieve model compression.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU training)

### Dependencies

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
Pillow>=9.0.0
wandb>=0.12.0 (optional, for logging)
```

## Installation

1. Clone the repository

2. Create a virtual environment

3. Install dependencies:
```bash
pip install torch torchvision numpy Pillow wandb
```

## Project Structure

```
.
├── mobilenet_v2.py      # MobileNetV2 architecture
├── dataloader.py        # CIFAR-10 data loading with augmentation
├── quantize.py          # Quantization implementation
├── utils.py             # Utility functions
├── main.py              # Training script
├── test.py              # Quantization evaluation script
├── sweep.py             # Hyperparameter sweep script
├── checkpoints/         # Saved model checkpoints
├── data/                # CIFAR-10 dataset (auto-downloaded)
└── results/             # Sweep results
```

## Usage

### 1. Training Baseline Model

Train MobileNetV2 on CIFAR-10 with default settings:

```bash
python main.py --epochs 200 --batch_size 128 --lr 0.1
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--width_mult` | 1.0 | Width multiplier for MobileNetV2 |
| `--dropout` | 0.2 | Dropout rate |
| `--epochs` | 200 | Number of training epochs |
| `--batch_size` | 128 | Batch size |
| `--lr` | 0.1 | Initial learning rate |
| `--optimizer` | sgd | Optimizer (sgd/adam) |
| `--momentum` | 0.9 | SGD momentum |
| `--weight_decay` | 4e-5 | Weight decay |
| `--label_smoothing` | 0.1 | Label smoothing factor |
| `--scheduler` | cosine | LR scheduler (cosine/step/none) |
| `--seed` | 42 | Random seed |
| `--wandb` | False | Enable wandb logging |

#### Example with Custom Settings

```bash
python main.py \
  --epochs 300 \
  --batch_size 256 \
  --lr 0.05 \
  --weight_decay 1e-4 \
  --scheduler cosine \
  --wandb \
  --wandb_project my-project \
  --seed 42
```

### 2. Evaluate with Quantization

Test the trained model with different quantization bit-widths:

```bash
python test.py \
  --checkpoint ./checkpoints/mobilenetv2_cifar10_best.pth \
  --weight_bits 8 \
  --act_bits 8
```

#### Quantization Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | ./checkpoints/mobilenetv2_cifar10_best.pth | Path to trained model |
| `--weight_bits` | 8 | Weight quantization bits |
| `--act_bits` | 8 | Activation quantization bits |
| `--calibration_batches` | 100 | Batches for activation calibration |
| `--save_quantized` | False | Save quantized model |
| `--wandb` | False | Enable wandb logging |


### Applied to MobileNetV2

Quantization is applied to:
- ✅ All convolutional layers (standard and depthwise)
- ✅ All fully connected layers
- ✅ All ReLU6 activations
- ❌ BatchNorm layers (fused with conv during inference)
- ❌ First and last layers (often kept at FP32 for minimal accuracy loss)

## Results

### Expected Performance

With proper training, you should achieve:

| Configuration | Accuracy | Model Size | Compression |
|--------------|----------|------------|-------------|
| FP32 Baseline | ~93-94% | ~9 MB | 1.0x |
| W8A8 | ~93% | ~2.3 MB | ~4x |
| W6A8 | ~92.5% | ~1.7 MB | ~5.3x |
| W4A8 | ~91% | ~1.2 MB | ~7.5x |
| W4A4 | ~88-90% | ~1.2 MB | ~7.5x |

## Reproducibility

### Seed Configuration

All scripts use fixed seeds for reproducibility:

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Environment

Tested on:
- **OS**: Ubuntu 24.04
- **Python**: 3.13
- **PyTorch**: latest
- **CUDA**: 13.0
- **GPU**: NVIDIA RTX 4060 (8GB)

### Exact Commands for Reproduction

1. **Train baseline**:
```bash
python main.py --epochs 300 --batch_size 256 --lr 0.05 --weight_decay 1e-4 --scheduler cosine --wandb --wandb_project Systems4DL-A3 --wandb_name baseline2 --seed 42
```

2. **Evaluate 8-bit quantization**:
```bash
python test.py --weight_bits 8 --act_bits 8 --seed 42
```

3. **Run full sweep**:
```bash
python sweep.py --checkpoint ./checkpoints/mobilenetv2_cifar10_best.pth --weight_bits_list 2 3 4 5 6 --act_bits_list 4 6 8 --per_channel_list True False --wandb --wandb_project WANDB_PROJECT_NAME
```
