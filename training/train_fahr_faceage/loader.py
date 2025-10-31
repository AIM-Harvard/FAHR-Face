import os
# moved from training/train_fahr_fageage to training/train_fahr_faceage
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np


def extract_age_from_path(path, dataset_type):
    """
    Function to extract age from the file path based on the dataset type.

    Args:
        path (str): Path to the image file.
        dataset_type (str): Type of the dataset (e.g., 'FGNET', 'MORPH', 'CACD', 'UTK', 'AFAD', 'IMDB_WIKI', 'AgeDB', 'MegaAge', 'KANFace').

    Returns:
        int: Extracted age from the file path.
    """
    base_name = os.path.basename(path)
    if dataset_type == "FGNET":
        age_part = base_name.split('A')[-1].split('.')[0]
        age = ''.join(filter(str.isdigit, age_part))
        return int(age)
    elif dataset_type == "MORPH":
        return int(base_name[-6:-4])
    elif dataset_type == "CACD":
        return int(base_name.split('_')[0])
    elif dataset_type == "UTK":
        age_str = base_name.split('_')[0]
        return int(age_str) if len(age_str) <= 2 else int(age_str[:3])
    elif dataset_type == "AFAD":
        # Navigate up two levels to get the age directory
        age_dir = os.path.dirname(os.path.dirname(path))
        age = os.path.basename(age_dir)
        return int(age)
    elif dataset_type == "IMDB_WIKI":
        parts = base_name.split('_')
        dob = parts[-2]
        photo_taken_year = parts[-1].split('.')[0]
        return int(photo_taken_year) - int(dob.split('-')[0])
    elif dataset_type == "AgeDB":
        age_part = base_name.split('A_')[-1].split('.')[0]
        return int(age_part)
    elif dataset_type == "MegaAge":
        age_part = base_name.split('_')[-1].split('.')[0]
        return int(age_part)
    elif dataset_type == "KANFace":
        parts = base_name.split('_')
        age_part = parts[1].split('(')[0]  # Stop at the parentheses or ignore them
        age = ''.join(filter(str.isdigit, age_part))
        return int(age)
    elif dataset_type == "LAG":
        age_part = base_name.split('_')[-1].split('.')[0]
        return int(age_part)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def oversample_data(image_paths, ages, dataset_types):
    """
    Oversample images of individuals based on the provided age distribution.

    Args:
        - image_paths (list): List of paths to the images.
        - ages (list): List of age labels corresponding to the images.
        - dataset_types (list): List of dataset types corresponding to the images.

    Returns:
        - oversampled_images (list): The list of image paths after oversampling.
        - oversampled_ages (list): The list of age labels after oversampling.
        - oversampled_dataset_types (list): The list of dataset types after oversampling.
    """
    distribution_factors = {

         (0, 50): 1,  # Begin moderate oversampling
         (50, 55): 2,
         (55, 60): 4,
         (60, 65): 6,
         (65, 70): 8,  # Increase oversampling as data becomes sparser
         (70, 75): 10,
         (75, 80): 12,  # More aggressive oversampling for very rare age groups
         (80, 85): 14,
         (85, 90): 16,
         (90, 95): 18,
         (95, 116):20,
         }


    oversampled = []
    for img_path, age, dataset_type in zip(image_paths, ages, dataset_types):
        factor = 1  # Default factor
        for age_range, f in distribution_factors.items():
            if age_range[0] <= age < age_range[1]:
                factor = f
                break
        oversampled.extend([(img_path, age, dataset_type)] * factor)

    oversampled_images, oversampled_ages, oversampled_dataset_types = zip(*oversampled)
    return list(oversampled_images), list(oversampled_ages), list(oversampled_dataset_types)


import random

def bin_and_balance_data(image_paths, ages, dataset_types, bin_size=200):
    """
    Bin ages into 10-year intervals and balance each bin to a specified size.

    Args:
        - image_paths (list): List of paths to the images.
        - ages (list): List of age labels corresponding to the images.
        - dataset_types (list): List of dataset types corresponding to the images.
        - bin_size (int): The desired size of each age bin (default: 2000).

    Returns:
        - balanced_images (list): The list of image paths after binning and balancing.
        - balanced_ages (list): The list of age labels after binning and balancing.
        - balanced_dataset_types (list): The list of dataset types after binning and balancing.
    """
    age_bins = [(i, i+5) for i in range(1,116,5)] + [(116, float('inf'))]
    binned_data = {bin_range: [] for bin_range in age_bins}

    for img_path, age, dataset_type in zip(image_paths, ages, dataset_types):
        for bin_range in age_bins:
            if bin_range[0] <= age < bin_range[1]:
                binned_data[bin_range].append((img_path, age, dataset_type))
                break

    balanced_data = []
    for bin_range, data in binned_data.items():
        if len(data) > bin_size:
            balanced_data.extend(random.sample(data, bin_size))
        else:
            balanced_data.extend(data)
            remaining = bin_size - len(data)
            balanced_data.extend(random.choices(data, k=remaining))

    random.shuffle(balanced_data)
    balanced_images, balanced_ages, balanced_dataset_types = zip(*balanced_data)
    return list(balanced_images), list(balanced_ages), list(balanced_dataset_types)



class AgeDataset(Dataset):
    def __init__(self, image_paths, ages, dataset_types, transform=None, min_age=1, max_age=116):
        self.transform = transform
        self.min_age = min_age
        self.max_age = max_age

        self.image_paths, self.ages, self.dataset_types = self._load_data(image_paths, ages, dataset_types)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        age = self.ages[idx]
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)
        return image, torch.tensor(age)

    def _load_data(self, image_paths, ages, dataset_types):
        filtered_image_paths = []
        filtered_ages = []
        filtered_dataset_types = []

        for img_path, age, dataset_type in zip(image_paths, ages, dataset_types):
            try:
                if self.min_age <= age <= self.max_age:
                    filtered_image_paths.append(img_path)
                    filtered_ages.append(age)
                    filtered_dataset_types.append(dataset_type)
            except ValueError as e:
                print(f"Error extracting age from path: {img_path}")
                print(f"Error message: {str(e)}")

        return filtered_image_paths, filtered_ages, filtered_dataset_types



train_transform = transforms.Compose([
     transforms.RandomResizedCrop(size=(112, 112), scale=(0.8, 1.0)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
     transforms.RandomGrayscale(p=0.2),
     transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.5),
     transforms.RandomRotation(degrees=10),
     transforms.ToTensor(),
     transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])





val_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) 
