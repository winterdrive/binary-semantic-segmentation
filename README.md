# Binary Semantic Segmentation with U-Net

A PyTorch implementation of U-Net for binary semantic segmentation on the Oxford-IIIT Pet Dataset. This project demonstrates end-to-end training and evaluation of deep learning models for computer vision tasks.

## Overview

This project implements a U-Net architecture for binary semantic segmentation, specifically designed to segment pets (cats and dogs) from background in images. The model learns to generate pixel-wise binary masks that distinguish between foreground (pet) and background regions.

### Key Features

- **U-Net Architecture**: Classic encoder-decoder network with skip connections
- **Binary Segmentation**: Optimized for foreground/background classification
- **Oxford-IIIT Pet Dataset**: Automatic dataset download and preprocessing
- **Training Pipeline**: Complete training loop with validation and logging
- **Evaluation Metrics**: IoU, accuracy, and visual result comparison
- **Model Checkpointing**: Save and load trained models

## Technical Architecture

### Model Architecture

- **Network**: U-Net with encoder-decoder structure
- **Input**: RGB images (3 channels, 256×256 pixels)
- **Output**: Binary masks (1 channel, 256×256 pixels)
- **Loss Function**: Binary Cross-Entropy with Logits
- **Optimizer**: Adam with learning rate scheduling

### Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.21.0
- Pillow >= 8.3.0
- tqdm >= 4.62.0
- matplotlib >= 3.4.0

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended) or CPU

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd binary-semantic-segmentation
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset**

   ```bash
   python -c "from src.dataset import OxfordPetDataset; OxfordPetDataset.download('./data/oxford-iiit-pet')"
   ```

## Usage

### Training

Train a U-Net model on the Oxford-IIIT Pet dataset:

```bash
python src/train.py --model_type unet --data_path ./data/oxford-iiit-pet --epochs 50 --batch_size 8 --learning_rate 1e-4
```

Train a ResNet34-UNet model:

```bash
python src/train.py --model_type resnet34_unet --data_path ./data/oxford-iiit-pet --epochs 50 --batch_size 8 --learning_rate 1e-4
```

#### Training Arguments

- `--model_type`: Type of model to train (`unet` or `resnet34_unet`)
- `--data_path`: Path to dataset directory (default: `./data/oxford-iiit-pet`)
- `--save_path`: Directory to save trained models (default: `./saved_models`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 8)
- `--learning_rate`: Learning rate for optimizer (default: 1e-4)

### Evaluation

Evaluate a trained U-Net model:

```bash
python src/evaluate.py --model_path ./saved_models/unet_best_model.pth --model_type unet --data_path ./data/oxford-iiit-pet --save_visualizations
```

Evaluate a trained ResNet34-UNet model:

```bash
python src/evaluate.py --model_path ./saved_models/resnet34_unet_best_model.pth --model_type resnet34_unet --data_path ./data/oxford-iiit-pet --save_visualizations
```

### Inference

Run inference with a trained U-Net model:

```bash
python src/inference.py --model ./saved_models/unet_best_model.pth --model_type unet --data_path ./data/oxford-iiit-pet --save_results
```

Run inference with a trained ResNet34-UNet model:

```bash
python src/inference.py --model ./saved_models/resnet34_unet_best_model.pth --model_type resnet34_unet --data_path ./data/oxford-iiit-pet --save_results
```

### Single Image Demo

Run inference on a single image:

```bash
python src/inference_demo.py --model_path ./saved_models/unet_best_model.pth --model_type unet --image_path ./demo/sample.jpg --output_path ./results/demo_result.png
```

### Quick Demo

Run the complete demo script:

```bash
cd demo && chmod +x demo.sh && ./demo.sh
```

## Project Structure

```text
binary-semantic-segmentation/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py           # U-Net model implementation
│   │   └── resnet34_unet.py  # ResNet34-UNet model implementation
│   ├── oxford_pet.py         # Oxford-IIIT Pet dataset loading and preprocessing
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── inference.py          # Inference script
│   └── utils.py              # Utility functions
├── demo/                     # Demo scripts and examples
├── data/                     # Dataset directory (created automatically)
├── requirements.txt          # Python dependencies
├── TECHNICAL_REPORT.md       # Detailed technical implementation report
└── README.md                # This file
```

### Key Files

- **`src/models/unet.py`**: U-Net architecture implementation with encoder-decoder structure
- **`src/models/resnet34_unet.py`**: ResNet34-UNet hybrid architecture implementation
- **`src/oxford_pet.py`**: Oxford-IIIT Pet dataset class with automatic download and preprocessing
- **`src/train.py`**: Complete training pipeline with validation and checkpointing
- **`src/evaluate.py`**: Model evaluation with metrics calculation and visualization
- **`src/utils.py`**: Helper functions for device selection, logging, and visualization
- **`TECHNICAL_REPORT.md`**: Comprehensive technical report with implementation details and experimental results

## Training Process

The training process includes:

1. **Data Loading**: Automatic dataset download and train/validation split (90%/10%)
2. **Preprocessing**: Image resizing to 256×256 and trimap conversion to binary masks
3. **Training Loop**: Forward pass, loss calculation, backpropagation with gradient clipping
4. **Validation**: Periodic evaluation on validation set with IoU and accuracy metrics
5. **Checkpointing**: Automatic saving of best models and periodic checkpoints
6. **Learning Rate Scheduling**: Adaptive learning rate reduction based on validation performance

## Expected Results

### Performance Metrics

- **Training Accuracy**: ~95%+ on training set
- **Validation Accuracy**: ~90%+ on validation set  
- **IoU Score**: ~0.8+ for well-segmented images
- **Convergence**: Typically converges within 30-50 epochs

### Model Performance

- **Inference Speed**: ~50-100ms per image on GPU
- **Model Size**: ~31M parameters (~120MB file size)
- **Memory Usage**: ~2-4GB GPU memory during training

For detailed experimental results, training insights, and comprehensive technical analysis, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Dataset Information

The **Oxford-IIIT Pet Dataset** contains:

- **37 pet categories** (cats and dogs)
- **~7,400 images** total
- **Trimap annotations** with pixel-level labels:
  - Class 1: Foreground (pet)
  - Class 2: Background
  - Class 3: Boundary/uncertain regions

The dataset is automatically downloaded and processed into binary masks suitable for semantic segmentation.

## References

- **U-Net Paper**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- **Oxford-IIIT Pet Dataset**: [Cats and Dogs Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- **PyTorch**: [Deep Learning Framework](https://pytorch.org/)

## License

This project is open source and available under the MIT License.
