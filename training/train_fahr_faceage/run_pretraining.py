import torch
# moved from training/train_fahr_fageage to training/train_fahr_faceage
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTConfig
from optimizer import create_optimizer
import os
from loader import AgeDataset, train_transform, val_transform, extract_age_from_path, oversample_data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ViTForAgeEstimation import ViTForAgeRegression, AgeRegressionOutput

checkpoint_path = "../pretraining/checkpoints/checkpoint_epoch_23.pth"
dataset_dirs = {
    "FGNET": "/path/to/FGNET_cropped_aligned/",
    "MORPH": "/path/to/morph_cropped_aligned/",
    "CACD": "/path/to/cacd_cropped_aligned/",
    "UTK": "/path/to/utk_cropped_aligned/",
    "AFAD": "/path/to/afad_cropped_original/",
    "IMDB_WIKI": "/path/to/imdb_cropped_aligned/",
    "AgeDB": "/path/to/AgeDB_cropped_aligned/",
    "MegaAge": "/path/to/megaage_cropped_aligned/",
    "KANFace": "/path/to/KANFace_cropped/",
    "LAG": "/path/to/lag/"
}
selected_datasets = ["IMDB_WIKI","KANFace","FGNET","CACD","AFAD","MegaAge","MORPH","LAG"]

learning_rate = 1e-5
epochs = 30
batch_size = 64
num_workers = 16

def load_dataset_images(dataset_dirs):
    all_images = []
    for dataset_name, dataset_path in dataset_dirs.items():
        for subdir, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subdir, file)
                    all_images.append((img_path, dataset_name))
    return all_images

def initialize_model(checkpoint_path, device):
    """Initialize and return the Vision Transformer model with pretrained weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ViTConfig(image_size=112, num_hidden_layers=12, num_attention_heads=12, hidden_dropout_prob=0.1,attention_probs_dropout_prob=0.1)
    model = ViTForAgeRegression(config)
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model.state_dict()}
    model.load_state_dict({**model.state_dict(), **pretrained_dict})
    return model.to(device)

#def save_checkpoint(model, optimizer, epoch, save_dir="checkpoints_pretraining"):
def save_checkpoint(model, optimizer, epoch, save_dir="checkpoints_pretraining"):
    """Save the current model and optimizer states as a checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)
    print(f"Checkpoint saved: {save_path}")

def adjust_learning_rate(optimizer, lr):
    """Adjusts the learning rate while preserving layer-wise scaling."""
    for param_group in optimizer.param_groups:
        scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = lr * scale

def train(model, train_loader, val_loader, optimizer, device, epochs, lr):
    loss_fn = nn.L1Loss()

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, lr)
        model.train()
        total_train_loss = 0.0

        print(f'Epoch {epoch + 1}/{epochs} [Training]')
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            age_predictions = outputs.age_output

            loss = loss_fn(age_predictions, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        print(f'Epoch {epoch + 1}/{epochs} [Validation]')
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                age_predictions = outputs.age_output

                loss = loss_fn(age_predictions, labels)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        save_checkpoint(model, optimizer, epoch + 1)
        print(f"Epoch {epoch + 1}, Train Loss (MAE): {avg_train_loss:.3f}, Validation Loss (MAE): {avg_val_loss:.3f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_dataset_dirs = {k: v for k, v in dataset_dirs.items() if k in selected_datasets}
    all_images = load_dataset_images(selected_dataset_dirs)
    all_image_paths, all_dataset_types = zip(*all_images)
    all_ages = [extract_age_from_path(path, dataset_type) for path, dataset_type in zip(all_image_paths, all_dataset_types)]

    # Split the data into train and validation sets
    train_image_paths, val_image_paths, train_ages, val_ages, train_dataset_types, val_dataset_types = train_test_split(
        all_image_paths, all_ages, all_dataset_types, test_size=0.02, random_state=42
    )

    # Apply oversampling only to the training data
    train_image_paths, train_ages, train_dataset_types = oversample_data(train_image_paths, train_ages, train_dataset_types)

    train_dataset = AgeDataset(train_image_paths, train_ages, train_dataset_types, transform=train_transform, min_age=1, max_age=116)
    val_dataset = AgeDataset(val_image_paths, val_ages, val_dataset_types, transform=val_transform, min_age=1, max_age=116)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = initialize_model(checkpoint_path, device)
    optimizer = create_optimizer(model, base_lr=learning_rate, weight_decay=0.05, layer_decay=0.75, beta1=0.9, beta2=0.999)
    train(model, train_loader, val_loader, optimizer, device, epochs, learning_rate)


if __name__ == "__main__":
    main()
