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

def train_quantization_model():
    """Main training function for WandB sweep."""
    # Initialize WandB run
    wandb.init()
    config = wandb.config
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Load dataset
    train_loader, test_loader = get_cifar10(batchsize=config.batch_size)
    
    # Load pre-trained model
    model = mobilenet_v2(num_classes=10, width_mult=config.width_mult, dropout=config.dropout)
    checkpoint = torch.load(config.checkpoint, map_location='cpu', weights_only=True)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Baseline accuracy
    baseline_acc = evaluate(model, test_loader, device)
    
    # Apply advanced quantization with sweep parameters
    advanced_swap_to_quant_modules(
        model, 
        weight_bits=config.weight_bits, 
        act_bits=config.act_bits,
        activations_unsigned=True,
        per_channel_weights=config.per_channel,
        weight_symmetric=config.weight_symmetric,
        activation_calibration=config.activation_calibration
    )
    model = model.to(device)
    
    # Enhanced calibration
    advanced_calibrate_model(model, train_loader, device, num_batches=config.calibration_batches)
    freeze_all_quant(model)
    
    # Evaluate quantized model
    quantized_acc = evaluate(model, test_loader, device)
    
    # Calculate comprehensive compression statistics
    compression_stats = calculate_compression_ratios(model, config.weight_bits, config.act_bits)
    
    # Calculate metrics for optimization
    accuracy_drop = baseline_acc - quantized_acc
    compression_accuracy_score = compression_stats['overall_compression'] / (1 + accuracy_drop)
    
    # Log all metrics to WandB
    metrics = {
        'baseline_accuracy': baseline_acc,
        'quantized_accuracy': quantized_acc,
        'accuracy_drop': accuracy_drop,
        'relative_accuracy_drop': (accuracy_drop / baseline_acc) * 100,
        'weight_compression_ratio': compression_stats['weight_compression'],
        'activation_compression_ratio': compression_stats['activation_compression'],
        'overall_compression_ratio': compression_stats['overall_compression'],
        'final_model_size_mb': compression_stats['quant_size_mb'],
        'model_size_reduction_percent': compression_stats['model_size_reduction'],
        'compression_accuracy_score': compression_accuracy_score,
        'efficiency_score': quantized_acc * np.log(compression_stats['overall_compression']) / 100
    }
    
    wandb.log(metrics)
    
    # Print results for this run
    print(f"\nRun completed:")
    print(f"  Config: W{config.weight_bits}A{config.act_bits}, "
          f"PerChannel:{config.per_channel}, Symmetric:{config.weight_symmetric}, "
          f"Calib:{config.activation_calibration}")
    print(f"  Results: Acc={quantized_acc:.2f}%, Drop={accuracy_drop:.2f}%, "
          f"Compression={compression_stats['overall_compression']:.2f}x")
    
    return quantized_acc

def setup_wandb_sweep(args):
    """Setup WandB sweep configuration for parallel coordinates plot."""
    
    sweep_config = {
        'method': 'grid',  # Use 'grid' for exhaustive search, 'random' for random search
        'metric': {
            'name': 'compression_accuracy_score',
            'goal': 'maximize'
        },
        'parameters': {
            'weight_bits': {
                'values': args.weight_bits_list
            },
            'act_bits': {
                'values': args.act_bits_list
            },
            'per_channel': {
                'values': args.per_channel_list
            },
            'weight_symmetric': {
                'values': args.weight_symmetric_list
            },
            'activation_calibration': {
                'values': args.calibration_methods
            }
        }
    }
    
    # Add fixed parameters
    sweep_config['parameters'].update({
        'checkpoint': {
            'value': args.checkpoint
        },
        'width_mult': {
            'value': args.width_mult
        },
        'dropout': {
            'value': args.dropout
        },
        'batch_size': {
            'value': args.batch_size
        },
        'calibration_batches': {
            'value': args.calibration_batches
        },
        'seed': {
            'value': args.seed
        }
    })
    
    return sweep_config

def run_single_evaluation(args):
    """Run a single evaluation with given parameters (for testing)."""
    set_seed(args.seed)
    
    # Load dataset
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10(batchsize=args.batch_size)
    
    # Load model
    model = mobilenet_v2(num_classes=10, width_mult=args.width_mult, dropout=args.dropout)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Baseline accuracy
    baseline_acc = evaluate(model, test_loader, device)
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    
    # Apply quantization
    advanced_swap_to_quant_modules(
        model, 
        weight_bits=args.weight_bits, 
        act_bits=args.act_bits,
        activations_unsigned=True,
        per_channel_weights=args.per_channel,
        weight_symmetric=args.weight_symmetric,
        activation_calibration=args.activation_calibration
    )
    model = model.to(device)
    
    # Calibrate and freeze
    advanced_calibrate_model(model, train_loader, device, num_batches=args.calibration_batches)
    freeze_all_quant(model)
    
    # Evaluate quantized model
    quantized_acc = evaluate(model, test_loader, device)
    
    # Calculate compression statistics
    compression_stats = calculate_compression_ratios(model, args.weight_bits, args.act_bits)
    
    print(f"\nQuantization Results:")
    print(f"Quantized Accuracy: {quantized_acc:.2f}%")
    print(f"Accuracy Drop: {baseline_acc - quantized_acc:.2f}%")
    print(f"Overall Compression: {compression_stats['overall_compression']:.2f}x")
    print(f"Final Model Size: {compression_stats['quant_size_mb']:.2f} MB")
    
    return {
        'weight_bits': args.weight_bits,
        'act_bits': args.act_bits,
        'per_channel': args.per_channel,
        'weight_symmetric': args.weight_symmetric,
        'activation_calibration': args.activation_calibration,
        'baseline_accuracy': baseline_acc,
        'quantized_accuracy': quantized_acc,
        'accuracy_drop': baseline_acc - quantized_acc,
        **compression_stats
    }

def main(args):
    """Main function to run either sweep or single evaluation."""
    
    if args.sweep_mode:
        # Run WandB sweep
        sweep_config = setup_wandb_sweep(args)
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        
        print(f"Starting WandB sweep with ID: {sweep_id}")
        print("Sweep configuration:")
        print(json.dumps(sweep_config, indent=2))
        
        # Run the sweep agent
        wandb.agent(sweep_id, train_quantization_model, count=args.sweep_count)
        
    else:
        # Run single evaluation
        if args.wandb:
            wandb.init(project=args.wandb_project, config=vars(args))
        
        result = run_single_evaluation(args)
        
        if args.wandb:
            wandb.log(result)
            wandb.finish()
        
        # Save single result
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, 'single_evaluation.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {result_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced quantization with WandB sweep')
    
    # Sweep mode selection
    parser.add_argument('--sweep_mode', action='store_true',
                       help='Run WandB sweep (otherwise single evaluation)')
    parser.add_argument('--sweep_count', type=int, default=50,
                       help='Number of sweep runs to execute')
    
    # Model parameters
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Quantization parameters (for single run or sweep bounds)
    parser.add_argument('--weight_bits', type=int, default=8,
                       help='Weight bits for single evaluation')
    parser.add_argument('--act_bits', type=int, default=8,
                       help='Activation bits for single evaluation')
    parser.add_argument('--per_channel', type=bool, default=True,
                       help='Per-channel quantization for single evaluation')
    parser.add_argument('--weight_symmetric', type=bool, default=True,
                       help='Symmetric quantization for single evaluation')
    parser.add_argument('--activation_calibration', type=str, default='minmax',
                       choices=['minmax', 'moving_average', 'percentile'],
                       help='Calibration method for single evaluation')
    
    # Sweep parameter ranges
    parser.add_argument('--weight_bits_list', type=int, nargs='+',
                       default=[2, 3, 4, 5, 6, 8],
                       help='Weight bit values for sweep')
    parser.add_argument('--act_bits_list', type=int, nargs='+',
                       default=[4, 6, 8],
                       help='Activation bit values for sweep')
    parser.add_argument('--per_channel_list', type=bool, nargs='+',
                       default=[True, False],
                       help='Per-channel values for sweep')
    parser.add_argument('--weight_symmetric_list', type=bool, nargs='+',
                       default=[True, False],
                       help='Symmetric quantization values for sweep')
    parser.add_argument('--calibration_methods', type=str, nargs='+',
                       default=['minmax', 'moving_average'],
                       help='Calibration methods for sweep')
    
    # Calibration parameters
    parser.add_argument('--calibration_batches', type=int, default=100)
    
    # Data
    parser.add_argument('--batch_size', type=int, default=128)
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./quantization_results')
    
    # Wandb
    parser.add_argument('--wandb', action='store_true',
                       help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='mobilenetv2-quantization-sweep')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    main(args)
