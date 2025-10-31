import pandas as pd
# moved from training/train_fahr_face_survivel to training/train_fahr_facesurvival
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SurvivalDataset(Dataset):
    def __init__(self, csv_file, photo_folder, transform=None):
        self.patient_data = pd.read_csv(csv_file, parse_dates=['death_or_followup_date', 'image_acquisition_date'])
        self.patient_data['survival_time_years'] = (self.patient_data['death_or_followup_date'] - self.patient_data['image_acquisition_date']).dt.days / 365
        self.photo_folder = photo_folder
        self.transform = transform

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx):
        patient_image_id = str(self.patient_data.iloc[idx]['patient_photo_id']) + '.jpg'
        survival_time_years = self.patient_data.iloc[idx]['survival_time_years']
        event_indicator = self.patient_data.iloc[idx]['censor']
        img_path = os.path.join(self.photo_folder, patient_image_id)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No photo file found for patient_photo_id {patient_image_id}")
        with Image.open(img_path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, torch.tensor(survival_time_years, dtype=torch.float), torch.tensor(event_indicator, dtype=torch.float)

train_transform = transforms.Compose([
     transforms.RandomResizedCrop(size=(112, 112), scale=(0.9, 1.0)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
     transforms.RandomRotation(degrees=5),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


val_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
