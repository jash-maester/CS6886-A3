import torch
import torch.nn as nn
import argparse
import os
from mobilenet_v2 import mobilenet_v2
from dataloader import get_cifar10
from utils import evaluate, set_seed
from quantize import (
    swap_to_quant_modules, 
    freeze_all_quant, 
    print_compression
)
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calibrate_model(model, data_loader, device, num_batches=100):
    """
    Calibrate the model by running inference on calibration data.
    This observes activation ranges for quantization.
    """
    model.eval()
    print(f"Calibrating model with {num_batches} batches...")
    
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            _ = model(x)
            if i >= num_batches:
                break
    print("Calibration completed!")

def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Initialize wandb if enabled
    if args.wandb:
        run_name = f"w{args.weight_bits}a{args.act_bits}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    
    # Load dataset
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10(batchsize=args.batch_size)
    
    # Create model
    print(f"Creating MobileNetV2 (width_mult={args.width_mult}, dropout={args.dropout})...")
    model = mobilenet_v2(num_classes=10, width_mult=args.width_mult, dropout=args.dropout)
    
    # Load trained weights
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Evaluate baseline model
    print("\n" + "="*60)
    print("BASELINE MODEL EVALUATION")
    print("="*60)
    baseline_acc = evaluate(model, test_loader, device)
    print(f"Baseline Test Accuracy: {baseline_acc:.2f}%")
    
    # Calculate baseline size
    baseline_size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
    print(f"Baseline Model Size: {baseline_size:.2f} MB (FP32)")
    print("="*60 + "\n")
    
    # Apply quantization
    print("="*60)
    print("APPLYING QUANTIZATION")
    print("="*60)
    print(f"Weight Quantization: {args.weight_bits} bits")
    print(f"Activation Quantization: {args.act_bits} bits")
    print("-"*60)
    
    swap_to_quant_modules(
        model, 
        weight_bits=args.weight_bits, 
        act_bits=args.act_bits,
        activations_unsigned=True
    )
    model = model.to(device)
    
    # Calibrate activations
    calibrate_model(model, train_loader, device, num_batches=args.calibration_batches)
    
    # Freeze quantization parameters
    print("Freezing quantization parameters...")
    freeze_all_quant(model)
    
    # Evaluate quantized model
    print("\n" + "="*60)
    print("QUANTIZED MODEL EVALUATION")
    print("="*60)
    quantized_acc = evaluate(model, test_loader, device)
    print(f"Quantized Test Accuracy: {quantized_acc:.2f}%")
    print(f"Accuracy Drop: {baseline_acc - quantized_acc:.2f}%")
    
    # Print compression statistics
    compression_stats = print_compression(model, weight_bits=args.weight_bits, act_bits=args.act_bits)
    
    # Log to wandb
    if args.wandb:
        wandb.log({
            'baseline_acc': baseline_acc,
            'quantized_acc': quantized_acc,
            'accuracy_drop': baseline_acc - quantized_acc,
            'weight_bits': args.weight_bits,
            'act_bits': args.act_bits,
            **compression_stats
        })
        wandb.finish()
    
    # Save quantized model if requested
    if args.save_quantized:
        save_path = os.path.join(
            args.checkpoint_dir,
            f'mobilenetv2_w{args.weight_bits}a{args.act_bits}.pth'
        )
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'quantized_acc': quantized_acc,
            'baseline_acc': baseline_acc,
            'weight_bits': args.weight_bits,
            'act_bits': args.act_bits,
        }, save_path)
        print(f"\nQuantized model saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test MobileNetV2 with Quantization')
    
    # Model parameters
    parser.add_argument('--width_mult', type=float, default=1.0,
                       help='Width multiplier for MobileNetV2')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Quantization parameters
    parser.add_argument('--weight_bits', type=int, default=8,
                       help='Bits for weight quantization (default: 8)')
    parser.add_argument('--act_bits', type=int, default=8,
                       help='Bits for activation quantization (default: 8)')
    parser.add_argument('--calibration_batches', type=int, default=100,
                       help='Number of batches for calibration (default: 100)')
    
    # Data parameters
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, 
                       default='./checkpoints/mobilenetv2_cifar10_best.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save quantized models')
    parser.add_argument('--save_quantized', action='store_true',
                       help='Save quantized model')
    
    # Wandb
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, 
                       default='mobilenetv2-quantization',
                       help='Wandb project name')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    main(args)
