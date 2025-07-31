import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import create_csv_logger, write_to_csv, get_device, calculate_dice_score


def train_model(args):
    """
    Train a U-Net model for binary semantic segmentation.
    
    Args:
        args: Parsed command line arguments containing training configuration
    """
    # Determine device (GPU/CPU)
    device = get_device()
    print(f"Using device: {device}")

    # Check if dataset exists
    if not os.path.exists(os.path.join(args.data_path, "images")):
        print("Dataset not found. Please download the Oxford-IIIT Pet dataset.")
        print("You can download it using:")
        print("python -c \"from src.dataset import OxfordPetDataset; OxfordPetDataset.download('./data/oxford-iiit-pet')\"")
        return

    # Load datasets
    train_dataset = load_dataset(args.data_path, mode="train")
    val_dataset = load_dataset(args.data_path, mode="valid")
    
    print(f"Found {len(train_dataset)} training images")
    print(f"Found {len(val_dataset)} validation images")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Initialize model
    if args.model_type == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model_type == 'resnet34_unet':
        model = ResNet34_UNet(num_classes=1, input_channels=3).to(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model_type} model has {total_params:,} parameters")

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_path, f"{args.model_type}_train_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # CSV logger for training metrics
    csv_logger = create_csv_logger(
        os.path.join(save_dir, "training_log.csv"),
        ['epoch', 'train_loss', 'train_dice', 'val_loss', 'val_dice', 'learning_rate']
    )

    best_val_dice = 0.0
    
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Results will be saved to: {save_dir}")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        num_train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch in train_pbar:
            images = batch['image'].float().to(device)
            masks = batch['mask'].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                dice = calculate_dice_score(torch.sigmoid(outputs), masks)
                train_dice += dice
            
            train_loss += loss.item()
            num_train_batches += 1
            
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{dice:.4f}"
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            
            for batch in val_pbar:
                images = batch['image'].float().to(device)
                masks = batch['mask'].float().to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                dice = calculate_dice_score(torch.sigmoid(outputs), masks)
                val_dice += dice
                val_loss += loss.item()
                num_val_batches += 1
                
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{dice:.4f}"
                })

        # Calculate epoch metrics
        avg_train_loss = train_loss / num_train_batches
        avg_val_loss = val_loss / num_val_batches
        avg_train_dice = train_dice / num_train_batches
        avg_val_dice = val_dice / num_val_batches
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        write_to_csv(csv_logger, [
            epoch + 1, avg_train_loss, avg_train_dice, avg_val_loss, avg_val_dice, current_lr
        ])

        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Update learning rate scheduler
        scheduler.step(avg_val_dice)

        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': avg_val_dice,
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  New best model saved! Validation Dice: {avg_val_dice:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': avg_val_dice,
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    print(f"Training completed! Best validation Dice score: {best_val_dice:.4f}")
    print(f"Models saved in: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models for binary semantic segmentation")
    
    parser.add_argument("--data_path", type=str, default="./data/oxford-iiit-pet",
                        help="Path to the dataset")
    parser.add_argument("--save_path", type=str, default="./saved_models",
                        help="Path to save trained models")
    parser.add_argument("--model_type", type=str, choices=['unet', 'resnet34_unet'], 
                        default='unet', help="Type of model to train")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(args.save_path, exist_ok=True)
    
    train_model(args)
