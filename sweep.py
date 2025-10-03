import torch
import argparse
import os
import json
from mobilenet_v2 import mobilenet_v2
from dataloader import get_cifar10
from utils import evaluate, set_seed
from quantize import (
    swap_to_quant_modules,
    freeze_all_quant,
    model_size_bytes_fp32,
    model_size_bytes_quant,
    calculate_metadata_size
)
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calibrate_model(model, data_loader, device, num_batches=100):
    """Calibrate model for activation quantization."""
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            _ = model(x)
            if i >= num_batches:
                break

def evaluate_quantization(checkpoint_path, weight_bits, act_bits, 
                         train_loader, test_loader, device, args):
    """Evaluate a specific quantization configuration."""
    
    # Load model
    model = mobilenet_v2(num_classes=10, width_mult=args.width_mult, dropout=args.dropout)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Baseline accuracy
    baseline_acc = evaluate(model, test_loader, device)
    baseline_size = model_size_bytes_fp32(model) / (1024 * 1024)
    
    # Apply quantization
    swap_to_quant_modules(model, weight_bits=weight_bits, act_bits=act_bits, 
                         activations_unsigned=True)
    model = model.to(device)
    
    # Calibrate and freeze
    calibrate_model(model, train_loader, device, num_batches=args.calibration_batches)
    freeze_all_quant(model)
    
    # Evaluate quantized model
    quantized_acc = evaluate(model, test_loader, device)
    
    # Calculate sizes
    quant_size = model_size_bytes_quant(model, weight_bits) / (1024 * 1024)
    metadata_size = calculate_metadata_size(model) / 1024  # KB
    total_size = quant_size + metadata_size / 1024  # MB
    
    # Calculate compression ratios
    weight_params = sum(p.numel() for name, p in model.named_parameters() if "weight" in name)
    weight_fp32 = weight_params * 4 / (1024 * 1024)
    weight_quant = weight_params * weight_bits / 8 / (1024 * 1024)
    weight_compression = weight_fp32 / weight_quant
    overall_compression = baseline_size / total_size
    
    return {
        'weight_bits': weight_bits,
        'act_bits': act_bits,
        'baseline_acc': baseline_acc,
        'quantized_acc': quantized_acc,
        'accuracy_drop': baseline_acc - quantized_acc,
        'baseline_size_mb': baseline_size,
        'quantized_size_mb': total_size,
        'weight_compression': weight_compression,
        'overall_compression': overall_compression,
        'metadata_kb': metadata_size
    }

def main(args):
    set_seed(args.seed)
    
    # Initialize wandb
    if args.wandb:
        wandb.init(project=args.wandb_project, name="quantization_sweep")
    
    # Load dataset
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10(batchsize=args.batch_size)
    
    # Define sweep configurations
    bit_configs = []
    for w_bits in args.weight_bits_list:
        for a_bits in args.act_bits_list:
            bit_configs.append((w_bits, a_bits))
    
    print(f"\nRunning sweep over {len(bit_configs)} configurations...")
    print("="*60)
    
    results = []
    
    for i, (w_bits, a_bits) in enumerate(bit_configs, 1):
        print(f"\n[{i}/{len(bit_configs)}] Testing W{w_bits}A{a_bits}...")
        print("-"*60)
        
        result = evaluate_quantization(
            args.checkpoint, w_bits, a_bits,
            train_loader, test_loader, device, args
        )
        
        results.append(result)
        
        # Print results
        print(f"Baseline Acc:      {result['baseline_acc']:.2f}%")
        print(f"Quantized Acc:     {result['quantized_acc']:.2f}%")
        print(f"Accuracy Drop:     {result['accuracy_drop']:.2f}%")
        print(f"Weight Comp:       {result['weight_compression']:.2f}x")
        print(f"Overall Comp:      {result['overall_compression']:.2f}x")
        print(f"Size:              {result['quantized_size_mb']:.2f} MB")
        
        # Log to wandb
        if args.wandb:
            wandb.log(result)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'sweep_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("SWEEP COMPLETED")
    print("="*60)
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print("\nSUMMARY:")
    print("-"*60)
    best_acc = max(results, key=lambda x: x['quantized_acc'])
    best_comp = max(results, key=lambda x: x['overall_compression'])
    best_tradeoff = max(results, key=lambda x: x['quantized_acc'] / (1 + abs(x['accuracy_drop'])))
    
    print(f"\nBest Accuracy: W{best_acc['weight_bits']}A{best_acc['act_bits']} - {best_acc['quantized_acc']:.2f}%")
    print(f"Best Compression: W{best_comp['weight_bits']}A{best_comp['act_bits']} - {best_comp['overall_compression']:.2f}x")
    print(f"Best Tradeoff: W{best_tradeoff['weight_bits']}A{best_tradeoff['act_bits']} - "
          f"{best_tradeoff['quantized_acc']:.2f}% @ {best_tradeoff['overall_compression']:.2f}x")
    
    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sweep quantization configurations')
    
    # Model parameters
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Quantization sweep
    parser.add_argument('--weight_bits_list', type=int, nargs='+',
                       default=[2, 3, 4, 5, 6, 8],
                       help='List of weight bit-widths to test')
    parser.add_argument('--act_bits_list', type=int, nargs='+',
                       default=[4, 6, 8],
                       help='List of activation bit-widths to test')
    parser.add_argument('--calibration_batches', type=int, default=100)
    
    # Data
    parser.add_argument('--batch_size', type=int, default=128)
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str,
                       default='./checkpoints/mobilenetv2_cifar10_best.pth')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results')
    
    # Wandb
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='mobilenetv2-sweep')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
