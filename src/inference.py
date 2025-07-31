"""
Binary Semantic Segmentation Inference Script

This script performs inference on test data using trained U-Net or ResNet34-UNet models.
Supports saving visualization results and calculating Dice scores when ground truth is available.

Usage:
    python inference.py --model_type unet --model saved_models/unet_best.pth --data_path dataset/oxford-iiit-pet --save_results
    python inference.py --model_type resnet34_unet --model saved_models/resnet34_unet_best.pth --data_path dataset/oxford-iiit-pet
"""

import argparse
import csv
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score, create_csv_logger, get_device
from tqdm import tqdm
from datetime import datetime
import time


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Perform inference on semantic segmentation models')
    parser.add_argument('--model', required=True, help='Path to the trained model weights')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the test dataset')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--model_type', type=str, choices=['unet', 'resnet34_unet'], required=True, 
                       help='Type of model to use for inference')
    parser.add_argument('--save_results', action='store_true', 
                       help='Save visualization results as images')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Directory to save inference results')
    return parser.parse_args()


def visualize_results(image, prediction, ground_truth=None, save_path=None, dice_score_val=None):
    """
    Visualize inference results with original image, prediction, and ground truth.
    
    Args:
        image (numpy.ndarray): Original input image
        prediction (numpy.ndarray): Model prediction mask
        ground_truth (numpy.ndarray, optional): Ground truth mask
        save_path (str, optional): Path to save the visualization
        dice_score_val (float, optional): Dice score to display
    """
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    # Normalize image for display
    image_display = (image - image.min()) / (image.max() - image.min())
    plt.imshow(np.transpose(image_display, (1, 2, 0)))
    plt.title("Original Image")
    plt.axis("off")
    
    # Predicted Mask
    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")
    
    # Ground Truth Mask (if available)
    plt.subplot(1, 3, 3)
    if ground_truth is not None:
        plt.imshow(ground_truth, cmap='gray')
        plt.title("Ground Truth Mask")
    else:
        plt.text(0.5, 0.5, 'No Ground Truth', ha='center', va='center', 
                transform=plt.gca().transAxes)
        plt.title("Ground Truth (Not Available)")
    plt.axis("off")
    
    # Add Dice score if available
    if dice_score_val is not None:
        plt.gcf().text(0.95, 0.95, f"Dice Score: {dice_score_val:.4f}", 
                      fontsize=12, color='red', ha='right', va='top',
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main inference function."""
    args = get_args()
    
    # Generate unique run ID
    run_id = int(time.time())
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {args.model_type} model from {args.model}")
    if args.model_type == 'unet':
        model = UNet(num_classes=1, input_channels=3).to(device)
    else:  # resnet34_unet
        model = ResNet34_UNet(num_classes=1, input_channels=3).to(device)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    model.eval()
    
    # Load test dataset
    print(f"Loading test dataset from {args.data_path}")
    try:
        dataset = load_dataset(args.data_path, mode="test")
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False)
        print(f"Test dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.model_type)
    if args.save_results:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    
    # Inference loop
    dice_total = 0.0
    count = 0
    
    print("Starting inference...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Processing", unit="batch")):
            # Get inputs
            images = batch['image'].float().to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            # Calculate Dice score if ground truth is available
            dice_val = None
            if "mask" in batch:
                masks = batch['mask'].float().to(device)
                dice_val = dice_score(predictions, masks).item()
                dice_total += dice_val
                count += 1
            
            # Save visualization if requested
            if args.save_results:
                # Convert to numpy for visualization
                image_np = images[0].cpu().numpy()
                pred_np = predictions[0].cpu().squeeze().numpy()
                gt_np = None
                if "mask" in batch:
                    gt_np = batch["mask"][0].cpu().squeeze().numpy()
                
                # Save result
                result_path = os.path.join(output_dir, f"result_{i:04d}.png")
                visualize_results(image_np, pred_np, gt_np, result_path, dice_val)
    
    # Calculate and display results
    if count > 0:
        avg_dice = dice_total / count
        print(f"\nInference completed!")
        print(f"Average Dice Score on test set: {avg_dice:.4f}")
        
        # Log results to CSV
        log_filepath = f"saved_models/{args.model_type}_inference_log.csv"
        log_header = ['type', 'model_type', 'run_id', 'epoch', 'batch', 'loss', 
                     'val_dice', 'dice_score', 'learning_rate', 'timestamp']
        
        # Create CSV logger if doesn't exist
        create_csv_logger(log_filepath, log_header)
        
        # Write inference results
        with open(log_filepath, "a", newline='') as log_file:
            csv_writer = csv.writer(log_file)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_data = ['inference', args.model_type, run_id, '', 0, '', '', 
                       avg_dice, '', timestamp]
            csv_writer.writerow(log_data)
            log_file.flush()
        
        print(f"Results logged to: {log_filepath}")
    else:
        print("\nInference completed!")
        print("No ground truth available for Dice score calculation.")
    
    if args.save_results:
        print(f"Visualization results saved to: {output_dir}")


if __name__ == '__main__':
    main()
