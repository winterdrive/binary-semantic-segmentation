"""
Oxford-IIIT Pet Dataset loading and preprocessing utilities.

This module provides dataset classes for loading and preprocessing the Oxford-IIIT Pet dataset
for binary semantic segmentation tasks.
"""

import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve


class OxfordPetDataset(torch.utils.data.Dataset):
    """
    Oxford-IIIT Pet Dataset class for binary semantic segmentation.

    Args:
        root (str): Root directory path of the dataset.
        mode (str): Dataset mode, choices: {"train", "valid", "test"}.
        transform (callable, optional): Data preprocessing or augmentation function.
    """

    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Get a single sample by index, including image, preprocessed mask and original trimap.
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: Dictionary containing 'image', 'mask', and 'trimap'
        """
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask, augment_uncertain=False):
        """
        Convert original trimap to binary mask.

        Original trimap labels:
            - 1 represents foreground, set to 1.0
            - 2 represents background, set to 0.0
            - 3 represents uncertain region, set to 1.0 or 0.0 based on augment_uncertain
            
        When augment_uncertain is True:
            Uncertain regions (3.0) are randomly set to foreground (1.0) or background (0.0) 
            with 50% probability each.
        When augment_uncertain is False:
            Both uncertain regions (3.0) and foreground (1.0) are treated as foreground (1.0).
            
        Args:
            mask (np.ndarray): Original trimap mask
            augment_uncertain (bool): Whether to randomly assign uncertain regions
            
        Returns:
            np.ndarray: Binary mask with values 0.0 or 1.0
        """
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        if augment_uncertain:
            uncertain = (mask == 3.0)
            rand = np.random.rand(*mask.shape)
            mask[uncertain] = np.where(rand[uncertain] < 0.5, 1.0, 0.0)
        else:
            mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        """
        Read file split information based on mode, return list of image filenames.
        
        Returns:
            list: List of image filenames for the specified mode
        """
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):
        """
        Download Oxford-IIIT Pet dataset to the specified root directory.
        
        Args:
            root (str): Root directory to download the dataset
        """
        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    """
    Simplified Oxford Pet Dataset class that inherits from OxfordPetDataset.
    
    Performs image and mask resizing and format conversion to prepare data for training.
    All images and masks are resized to 256x256 and converted from HWC to CHW format.
    """

    def __getitem__(self, *args, **kwargs):
        """
        Get preprocessed sample with resized images and format conversion.
        
        Returns:
            dict: Dictionary with preprocessed 'image', 'mask', and 'trimap'
                 - Images are resized to 256x256 and converted to CHW format
                 - Masks and trimaps are resized with nearest neighbor interpolation
        """
        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    """
    TqdmUpTo class for displaying download progress.
    
    Extends tqdm to show download progress with update_to method.
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update tqdm progress display.
        
        Args:
            b (int): Number of blocks transferred so far
            bsize (int): Size of each block
            tsize (int): Total size of the file
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    """
    Download file from given URL to specified filepath.
    
    If file already exists, skip download.
    
    Args:
        url (str): URL to download from
        filepath (str): Local path to save the file
    """
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    """
    Extract archive file to the same directory.
    
    Args:
        filepath (str): Path to the archive file
    """
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)


def load_dataset(data_path, mode):
    """
    Load dataset based on data path and mode.

    Args:
        data_path (str): Dataset path.
        mode (str): Mode, choices: "train", "valid", "test".

    Returns:
        SimpleOxfordPetDataset: Dataset instance for the specified mode.
    """
    return SimpleOxfordPetDataset(root=data_path, mode=mode)


# Demo function to show processed mask results  
if __name__ == "__main__":
    print("Oxford Pet Dataset module loaded successfully")
    print("Use load_dataset() to create train/test datasets")
