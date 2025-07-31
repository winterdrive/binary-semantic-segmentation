#!/bin/bash
# Binary Semantic Segmentation Demo Script

echo "==== Binary Semantic Segmentation Demo ===="

echo "==== 1. Installing Required Packages ===="
pip install -r ../requirements.txt

echo "==== 2. Checking and Downloading Oxford-IIIT Pet Dataset ===="
if [ ! -d "../data/oxford-iiit-pet/images" ]; then
    echo "Dataset not found, starting download..."
    python3 -c "from src.dataset import OxfordPetDataset; OxfordPetDataset.download('./data/oxford-iiit-pet')"
else
    echo "Dataset already exists, skipping download."
fi

echo "==== 3. Finding Latest Models ===="

# Find UNet model
LATEST_UNET_MODEL=$(find ../saved_models -name "*unet*best_model.pth" -type f | head -n 1)
if [ -z "$LATEST_UNET_MODEL" ]; then
    echo "No UNet model found. Please train a model first using:"
    echo "python ../src/train.py --model_type unet --data_path ../data/oxford-iiit-pet --epochs 50"
    echo "Training UNet model..."
    python3 ../src/train.py --model_type unet --data_path ../data/oxford-iiit-pet --epochs 10
    LATEST_UNET_MODEL=$(find ../saved_models -name "*unet*best_model.pth" -type f | head -n 1)
fi
echo "Using UNet model: $LATEST_UNET_MODEL"

# Find ResNet34-UNet model
LATEST_RESNET_MODEL=$(find ../saved_models -name "*resnet34_unet*best_model.pth" -type f | head -n 1)
if [ -z "$LATEST_RESNET_MODEL" ]; then
    echo "No ResNet34-UNet model found. Training ResNet34-UNet model..."
    python3 ../src/train.py --model_type resnet34_unet --data_path ../data/oxford-iiit-pet --epochs 10
    LATEST_RESNET_MODEL=$(find ../saved_models -name "*resnet34_unet*best_model.pth" -type f | head -n 1)
fi
echo "Using ResNet34-UNet model: $LATEST_RESNET_MODEL"

echo "==== 4. Running UNet Evaluation ===="
python3 ../src/evaluate.py \
    --model_path "$LATEST_UNET_MODEL" \
    --model_type unet \
    --data_path "../data/oxford-iiit-pet" \
    --output_dir "./results/unet_evaluation" \
    --save_visualizations

echo "==== 5. Running ResNet34-UNet Evaluation ===="
python3 ../src/evaluate.py \
    --model_path "$LATEST_RESNET_MODEL" \
    --model_type resnet34_unet \
    --data_path "../data/oxford-iiit-pet" \
    --output_dir "./results/resnet34_unet_evaluation" \
    --save_visualizations

echo "==== 6. Running UNet Inference ===="
python3 ../src/inference.py \
    --model "$LATEST_UNET_MODEL" \
    --model_type unet \
    --data_path "../data/oxford-iiit-pet" \
    --batch_size 8 \
    --save_results

echo "==== 7. Running ResNet34-UNet Inference ===="
python3 ../src/inference.py \
    --model "$LATEST_RESNET_MODEL" \
    --model_type resnet34_unet \
    --data_path "../data/oxford-iiit-pet" \
    --batch_size 8 \
    --save_results

echo "==== 8. Running Single Image Demo ===="
echo "Testing single image inference..."
python3 ../src/inference_demo.py \
    --image_path "../data/oxford-iiit-pet/images/Abyssinian_1.jpg" \
    --model_path "$LATEST_UNET_MODEL" \
    --model_type unet \
    --output_path "./results/single_demo_unet.png"

echo "==== Demo Complete! ===="
echo "Results saved in:"
echo "  - Evaluation results: ./results/"
echo "  - Inference results: ../inference_results/"
echo "  - Single image demo: ./results/single_demo_unet.png"
