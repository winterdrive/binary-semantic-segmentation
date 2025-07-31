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

echo "==== 3. Finding Latest UNet Model ===="
LATEST_UNET_MODEL=$(find ../saved_models -name "*unet*best_model.pth" -type f | head -n 1)
if [ -z "$LATEST_UNET_MODEL" ]; then
    echo "No UNet model found. Please train a model first using:"
    echo "python src/train.py --data_path ./data/oxford-iiit-pet --epochs 50"
    exit 1
fi
echo "Using UNet model: $LATEST_UNET_MODEL"

echo "==== 4. Running UNet Evaluation ===="
python3 ../src/evaluate.py \
    --model_path "$LATEST_UNET_MODEL" \
    --data_path "../data/oxford-iiit-pet" \
    --output_dir "./results/unet_evaluation"
fi
echo "ResNet34_UNet 權重: $LATEST_RESNET_MODEL"

echo "==== 5. 執行 UNet 推論並輸出 dice score ===="
python3 ../src/inference.py --model "$LATEST_UNET_MODEL" --data_path ../dataset/oxford-iiit-pet --model_type unet --batch_size 32 --save_results

echo "==== 6. 執行 ResNet34_UNet 推論並輸出 dice score ===="
python3 ../src/inference.py --model "$LATEST_RESNET_MODEL" --data_path ../dataset/oxford-iiit-pet --model_type resnet34_unet --batch_size 32 --save_results

echo "==== 7. 完成！===="
echo "推論結果已儲存於 ../inference_results/，dice score 會顯示於終端機並寫入 ../saved_models/unet_log.csv 及 ../saved_models/resnet34_unet_log.csv"