import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import (get_device, load_model_checkpoint, calculate_iou, 
                   calculate_dice_score, save_prediction_visualization)


def evaluate_model(args):
    """
    Evaluate a trained U-Net model on the test dataset.
    
    Args:
        args: Parsed command line arguments containing evaluation configuration
    """
    # Setup device
    device = get_device()
    
    # Load dataset
    if not os.path.exists(os.path.join(args.data_path, "images")):
        print("Dataset not found. Please download the Oxford-IIIT Pet dataset first.")
        return
    
    test_dataset = load_dataset(args.data_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Evaluating on {len(test_dataset)} test images")
    
    # Load model
    if args.model_type == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif args.model_type == 'resnet34_unet':
        model = ResNet34_UNet(num_classes=1, input_channels=3)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    load_model_checkpoint(model, args.model_path, device)
    model = model.to(device)
    model.eval()
    
    print(f"Evaluating {args.model_type} model")
    
    # Create output directory
    if args.save_visualizations:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluation metrics
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0
    
    print("Running evaluation...")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].float().to(device)
            masks = batch['mask'].float().to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            # Calculate metrics for each sample in the batch
            for i in range(images.shape[0]):
                pred_mask = predictions[i:i+1]
                true_mask = masks[i:i+1]
                
                iou = calculate_iou(pred_mask, true_mask)
                dice = calculate_dice_score(pred_mask, true_mask)
                
                total_iou += iou
                total_dice += dice
                num_samples += 1
                
                # Save visualization if requested
                if args.save_visualizations and batch_idx < args.max_visualizations:
                    image_np = images[i].cpu().numpy()
                    true_mask_np = true_mask[i].cpu().numpy()
                    pred_mask_np = (pred_mask[i] > 0.5).float().cpu().numpy()
                    
                    save_path = os.path.join(
                        args.output_dir, 
                        f"evaluation_sample_{batch_idx}_{i}.png"
                    )
                    save_prediction_visualization(
                        image_np, true_mask_np, pred_mask_np, save_path
                    )
            
            # Update progress bar
            avg_iou = total_iou / num_samples
            avg_dice = total_dice / num_samples
            pbar.set_postfix({
                'IoU': f"{avg_iou:.4f}",
                'Dice': f"{avg_dice:.4f}"
            })
    
    # Final results
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    
    print(f"\nEvaluation Results:")
    print(f"  Average IoU: {avg_iou:.4f}")
    print(f"  Average Dice Score: {avg_dice:.4f}")
    print(f"  Evaluated on {num_samples} samples")
    
    if args.save_visualizations:
        print(f"  Visualizations saved to: {args.output_dir}")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Average IoU: {avg_iou:.4f}\n")
        f.write(f"Average Dice Score: {avg_dice:.4f}\n")
    
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation models")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, choices=['unet', 'resnet34_unet'], 
                        required=True, help="Type of model to evaluate")
    parser.add_argument("--data_path", type=str, default="./data/oxford-iiit-pet",
                        help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save visualization images")
    parser.add_argument("--max_visualizations", type=int, default=20,
                        help="Maximum number of visualization images to save")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluate_model(args)
