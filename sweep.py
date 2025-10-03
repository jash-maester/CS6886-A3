import torch
import argparse
import os
import json
import numpy as np
from mobilenet_v2 import mobilenet_v2
from dataloader import get_cifar10
from utils import evaluate, set_seed
from quantize import (
    advanced_swap_to_quant_modules,
    freeze_all_quant,
    calculate_compression_ratios
)
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def advanced_calibrate_model(model, data_loader, device, num_batches=100):
    """Enhanced calibration with multiple passes for stability."""
    model.eval()
    
    # First pass: warm up calibration
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            _ = model(x)
            if i >= num_batches // 2:
                break
    
    # Second pass: final calibration
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            _ = model(x)
            if i >= num_batches // 2:
                break

def evaluate_advanced_quantization(checkpoint_path, weight_bits, act_bits, 
                                  train_loader, test_loader, device, args):
    """Evaluate advanced quantization configurations."""
    
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
    
    # Apply advanced quantization
    advanced_swap_to_quant_modules(
        model, 
        weight_bits=weight_bits, 
        act_bits=act_bits,
        activations_unsigned=True,
        per_channel_weights=args.per_channel,
        weight_symmetric=args.weight_symmetric,
        activation_calibration=args.activation_calibration
    )
    model = model.to(device)
    
    # Enhanced calibration
    advanced_calibrate_model(model, train_loader, device, num_batches=args.calibration_batches)
    freeze_all_quant(model)
    
    # Evaluate quantized model
    quantized_acc = evaluate(model, test_loader, device)
    
    # Calculate comprehensive compression statistics
    compression_stats = calculate_compression_ratios(model, weight_bits, act_bits)
    
    result = {
        'weight_bits': weight_bits,
        'act_bits': act_bits,
        'per_channel': args.per_channel,
        'weight_symmetric': args.weight_symmetric,
        'activation_calibration': args.activation_calibration,
        'baseline_acc': baseline_acc,
        'quantized_acc': quantized_acc,
        'accuracy_drop': baseline_acc - quantized_acc,
        'relative_accuracy_drop': (baseline_acc - quantized_acc) / baseline_acc * 100,
    }
    result.update(compression_stats)
    
    return result

def find_optimal_configuration(results):
    """Find optimal configurations based on different criteria."""
    if not results:
        return {}
    
    # Best accuracy
    best_acc = max(results, key=lambda x: x['quantized_acc'])
    
    # Best compression with reasonable accuracy (within 2% drop)
    reasonable_results = [r for r in results if r['accuracy_drop'] <= 2.0]
    if reasonable_results:
        best_comp_reasonable = max(reasonable_results, key=lambda x: x['overall_compression'])
    else:
        best_comp_reasonable = max(results, key=lambda x: x['overall_compression'])
    
    # Best tradeoff (accuracy vs compression)
    best_tradeoff = max(results, key=lambda x: x['quantized_acc'] / (1 + x['accuracy_drop']))
    
    # Best for very low bits
    low_bit_results = [r for r in results if r['weight_bits'] <= 4 and r['act_bits'] <= 4]
    if low_bit_results:
        best_low_bit = max(low_bit_results, key=lambda x: x['quantized_acc'])
    else:
        best_low_bit = min(results, key=lambda x: x['weight_bits'] + x['act_bits'])
    
    return {
        'best_accuracy': best_acc,
        'best_compression_reasonable': best_comp_reasonable,
        'best_tradeoff': best_tradeoff,
        'best_low_bit': best_low_bit
    }

def main(args):
    set_seed(args.seed)
    
    # Initialize wandb
    if args.wandb:
        wandb.init(project=args.wandb_project, name="advanced_quantization_sweep", config=vars(args))
    
    # Load dataset
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10(batchsize=args.batch_size)
    
    # Define advanced sweep configurations
    configurations = []
    
    # Test different combinations
    for w_bits in args.weight_bits_list:
        for a_bits in args.act_bits_list:
            for per_channel in args.per_channel_list:
                for weight_symmetric in args.weight_symmetric_list:
                    for calib in args.calibration_methods:
                        configurations.append({
                            'weight_bits': w_bits,
                            'act_bits': a_bits,
                            'per_channel': per_channel,
                            'weight_symmetric': weight_symmetric,
                            'activation_calibration': calib
                        })
    
    print(f"\nRunning advanced sweep over {len(configurations)} configurations...")
    print("="*80)
    
    results = []
    
    for i, config in enumerate(configurations, 1):
        print(f"\n[{i}/{len(configurations)}] Testing W{config['weight_bits']}A{config['act_bits']} "
              f"PerChannel:{config['per_channel']} Symmetric:{config['weight_symmetric']} "
              f"Calib:{config['activation_calibration']}")
        print("-"*80)
        
        # Update args with current configuration
        args.per_channel = config['per_channel']
        args.weight_symmetric = config['weight_symmetric']
        args.activation_calibration = config['activation_calibration']
        
        try:
            result = evaluate_advanced_quantization(
                args.checkpoint, config['weight_bits'], config['act_bits'],
                train_loader, test_loader, device, args
            )
            
            results.append(result)
            
            # Print results
            print(f"Baseline Acc:      {result['baseline_acc']:.2f}%")
            print(f"Quantized Acc:     {result['quantized_acc']:.2f}%")
            print(f"Accuracy Drop:     {result['accuracy_drop']:.2f}%")
            print(f"Weight Comp:       {result['weight_compression']:.2f}x")
            print(f"Overall Comp:      {result['overall_compression']:.2f}x")
            print(f"Size:              {result['quant_size_mb']:.2f} MB")
            
            # Log to wandb
            if args.wandb:
                wandb.log(result)
                
        except Exception as e:
            print(f"Error in configuration: {e}")
            continue
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'advanced_sweep_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find optimal configurations
    optimal_configs = find_optimal_configuration(results)
    
    print("\n" + "="*80)
    print("ADVANCED SWEEP COMPLETED")
    print("="*80)
    print(f"Results saved to: {results_path}")
    
    # Print comprehensive summary
    print("\nOPTIMAL CONFIGURATIONS SUMMARY:")
    print("="*80)
    
    for config_name, config in optimal_configs.items():
        if config:
            print(f"\n{config_name.replace('_', ' ').title()}:")
            print(f"  Configuration: W{config['weight_bits']}A{config['act_bits']} "
                  f"PerChannel:{config['per_channel']} Symmetric:{config['weight_symmetric']} "
                  f"Calib:{config['activation_calibration']}")
            print(f"  Quantized Accuracy: {config['quantized_acc']:.2f}%")
            print(f"  Accuracy Drop:      {config['accuracy_drop']:.2f}%")
            print(f"  Overall Compression:{config['overall_compression']:.2f}x")
            print(f"  Model Size:         {config['quant_size_mb']:.2f} MB")
    
    # Save optimal configurations
    optimal_path = os.path.join(args.output_dir, 'optimal_configurations.json')
    with open(optimal_path, 'w') as f:
        # Convert any tensor values to Python scalars for JSON serialization
        serializable_optimal = {}
        for k, v in optimal_configs.items():
            if v:
                serializable_optimal[k] = {key: (val.item() if torch.is_tensor(val) else val) 
                                         for key, val in v.items()}
        json.dump(serializable_optimal, f, indent=2)
    
    print(f"\nOptimal configurations saved to: {optimal_path}")
    
    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced quantization sweep')
    
    # Model parameters
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Advanced quantization options
    parser.add_argument('--weight_bits_list', type=int, nargs='+',
                       default=[2, 3, 4, 5, 6, 8],
                       help='List of weight bit-widths to test')
    parser.add_argument('--act_bits_list', type=int, nargs='+',
                       default=[4, 6, 8],
                       help='List of activation bit-widths to test')
    parser.add_argument('--per_channel_list', type=bool, nargs='+',
                       default=[True, False],
                       help='Test per-channel vs per-tensor quantization')
    parser.add_argument('--weight_symmetric_list', type=bool, nargs='+',
                       default=[True, False],
                       help='Test symmetric vs asymmetric weight quantization')
    parser.add_argument('--calibration_methods', type=str, nargs='+',
                       default=['minmax', 'moving_average'],
                       help='Activation calibration methods')
    
    # Calibration parameters
    parser.add_argument('--calibration_batches', type=int, default=100)
    
    # Data
    parser.add_argument('--batch_size', type=int, default=128)
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./advanced_results')
    
    # Wandb
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='mobilenetv2-advanced-sweep')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set default values for configuration parameters
    args.per_channel = True
    args.weight_symmetric = True
    args.activation_calibration = 'minmax'
    
    main(args)
