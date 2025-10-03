import torch
import torch.nn as nn
import argparse
import os
from mobilenet_v2 import mobilenet_v2
from dataloader import get_cifar10
from utils import evaluate, set_seed
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.3f}')
    
    train_acc = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    return avg_loss, train_acc

def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10(batchsize=args.batch_size)
    
    # Create model
    print(f"Creating MobileNetV2 (width_mult={args.width_mult}, dropout={args.dropout})...")
    model = mobilenet_v2(num_classes=10, width_mult=args.width_mult, dropout=args.dropout)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[args.epochs//2, 3*args.epochs//4], gamma=0.1
        )
    else:
        scheduler = None
    
    # Training loop
    best_acc = 0.0
    train_losses = []
    train_accs = []
    test_accs = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-"*60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_acc = evaluate(model, test_loader, device)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # Log to wandb
        if args.wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Print epoch summary
        print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f} | "
              f"Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'train_acc': train_acc,
            }, checkpoint_path)
            print(f"  → Best model saved! (Test Acc: {best_acc:.2f}%)")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, f'{args.model_name}_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'train_acc': train_acc,
    }, final_path)
    
    print("\n" + "="*60)
    print(f"Training completed!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print("="*60)
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MobileNetV2 on CIFAR-10')
    
    # Model parameters
    parser.add_argument('--width_mult', type=float, default=1.0,
                       help='Width multiplier for MobileNetV2 (default: 1.0)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (default: 0.2)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam'], help='Optimizer (default: sgd)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=4e-5,
                       help='Weight decay (default: 4e-5)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                       help='Use Nesterov momentum')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor (default: 0.1)')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler (default: cosine)')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--model_name', type=str, default='mobilenetv2_cifar10',
                       help='Model name for saving')
    
    # Wandb logging
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='mobilenetv2-cifar10',
                       help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default='baseline',
                       help='Wandb run name')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    main(args)
