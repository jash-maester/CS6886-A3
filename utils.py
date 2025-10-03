import torch
import torch.nn as nn
import numpy as np
import random

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, testloader, device):
    """
    Evaluate model accuracy on test set.
    
    Args:
        model: PyTorch model
        testloader: DataLoader for test data
        device: Device to run evaluation on
    
    Returns:
        Accuracy as percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy

def count_parameters(model):
    """Count total and trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_model_size_mb(model):
    """Calculate model size in MB (FP32)."""
    param_size = sum(p.numel() * 4 for p in model.parameters())
    buffer_size = sum(b.numel() * 4 for b in model.buffers())
    total_size = param_size + buffer_size
    return total_size / (1024 ** 2)
