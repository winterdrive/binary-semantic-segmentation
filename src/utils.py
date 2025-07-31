import csv
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_device():
    """
    Determine the best available device (CUDA GPU or CPU).
    
    Returns:
        torch.device: The device to use for computations
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    return device


def create_csv_logger(filepath, headers):
    """
    Create a CSV logger with specified headers.
    
    Args:
        filepath (str): Path to the CSV file
        headers (list): List of column headers
        
    Returns:
        file object: Opened CSV file object
    """
    csv_file = open(filepath, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(headers)
    return csv_file


def write_to_csv(csv_file, data):
    """
    Write data to CSV file.
    
    Args:
        csv_file: Opened CSV file object
        data (list): Data to write as a row
    """
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(data)
    csv_file.flush()


def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for binary masks.
    
    Args:
        pred_mask (torch.Tensor): Predicted mask
        true_mask (torch.Tensor): Ground truth mask
        threshold (float): Threshold for binarizing predictions
        
    Returns:
        float: IoU score
    """
    pred_binary = (pred_mask > threshold).float()
    true_binary = (true_mask > threshold).float()
    
    intersection = (pred_binary * true_binary).sum()
    union = pred_binary.sum() + true_binary.sum() - intersection
    
    if union == 0:
        return 1.0  # Perfect score when both masks are empty
    
    return (intersection / union).item()


def calculate_dice_score(pred_mask, true_mask, threshold=0.5):
    """
    Calculate Dice coefficient for binary masks.
    
    Args:
        pred_mask (torch.Tensor): Predicted mask
        true_mask (torch.Tensor): Ground truth mask
        threshold (float): Threshold for binarizing predictions
        
    Returns:
        float: Dice score
    """
    pred_binary = (pred_mask > threshold).float()
    true_binary = (true_mask > threshold).float()
    
    intersection = (pred_binary * true_binary).sum()
    total = pred_binary.sum() + true_binary.sum()
    
    if total == 0:
        return 1.0  # Perfect score when both masks are empty
    
    return (2.0 * intersection / total).item()


def save_prediction_visualization(image, true_mask, pred_mask, save_path):
    """
    Save a visualization comparing input image, ground truth, and prediction.
    
    Args:
        image (numpy.ndarray): Input image
        true_mask (numpy.ndarray): Ground truth mask
        pred_mask (numpy.ndarray): Predicted mask
        save_path (str): Path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if image.shape[0] == 3:  # CHW format
        image = np.transpose(image, (1, 2, 0))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Ground truth mask
    if true_mask.shape[0] == 1:  # CHW format
        true_mask = true_mask[0]
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    # Predicted mask
    if pred_mask.shape[0] == 1:  # CHW format
        pred_mask = pred_mask[0]
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_model_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path (str): Path to checkpoint file
        device: Device to load the model on
        
    Returns:
        dict: Checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_acc' in checkpoint:
        print(f"Checkpoint validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    return checkpoint


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_summary(model, input_shape=(3, 256, 256)):
    """
    Create a summary of model architecture and parameters.
    
    Args:
        model: PyTorch model
        input_shape (tuple): Shape of input tensor (C, H, W)
        
    Returns:
        str: Model summary string
    """
    total_params = count_parameters(model)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    summary = f"""
Model Summary:
- Total trainable parameters: {total_params:,}
- Model size: {model_size_mb:.2f} MB
- Input shape: {input_shape}
- Architecture: {model.__class__.__name__}
"""
    return summary
