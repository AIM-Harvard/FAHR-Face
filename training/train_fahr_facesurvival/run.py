import torch
# moved from training/train_fahr_face_survivel to training/train_fahr_facesurvival
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTConfig
from ViTForSurvivalAnalysis import ViTForSurvivalAnalysis
from loader import SurvivalDataset, train_transform, val_transform
from optimizer import create_optimizer
import numpy as np
import os
from concordance_index import c_index

# Hyperparameters
smooth_reg = 1e-4  # Smoothed regularization strength
epochs = 10
batch_size = 32
learning_rate = 1e-5

# Dataset paths
csv_file = 'training_data.csv'   # Path to the training data CSV file (photos used specifig for training)
photo_folder = "/path/to/facephotos"   # Path to the folder containing the face photos

# Path to the pretrained checkpoint
checkpoint_path = "/path/to/checkpoint"
# Device to run the model on (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RankingLoss(nn.Module):
    def __init__(self, smooth_reg=0.01):
        super(RankingLoss, self).__init__()
        self.smooth_reg = smooth_reg

    def forward(self, risk_scores, survival_times, censor):
        comparable_pairs = []
        for i in range(len(survival_times)):
            if censor[i] == 1:  # Consider only uncensored instances as the first instance in a pair
                for j in range(len(survival_times)):
                    if survival_times[j] > survival_times[i]:  # Include all instances with longer survival times
                        comparable_pairs.append((i, j))

        loss = 0
        for i, j in comparable_pairs:
            risk_diff = risk_scores[i] - risk_scores[j]
            loss += 1 - torch.sigmoid(risk_diff)

        if len(comparable_pairs) > 0:
            loss /= len(comparable_pairs)

        # Add smoothed regularization term
        if self.smooth_reg > 0:
            squared_diff = torch.square(risk_scores[1:] - risk_scores[:-1])
            smooth_term = self.smooth_reg * torch.mean(squared_diff)
            loss += smooth_term

        return loss


def initialize_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ViTForSurvivalAnalysis(ViTConfig(image_size=112, num_hidden_layers=12, num_attention_heads=12, hidden_dropout_prob=0.1))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and 'decoder' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)
    return model


def save_checkpoint(model, optimizer, epoch, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"Checkpoint saved: {save_path}")


def train(model, train_loader, val_loader, optimizer, device, epochs, smooth_reg):
    loss_fn = RankingLoss(smooth_reg)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_risk_scores = []
        train_survival_times = []
        train_censor = []

        for inputs, survival_times, censor in train_loader:
            inputs = inputs.to(device)
            survival_times = survival_times.to(device)
            censor = censor.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            risk_scores = outputs.logits.squeeze()  # Remove the extra dimension
            loss = loss_fn(risk_scores, survival_times, censor)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_risk_scores.extend(risk_scores.detach().cpu().numpy())
            train_survival_times.extend(survival_times.detach().cpu().numpy())
            train_censor.extend(censor.detach().cpu().numpy())

        train_loss /= len(train_loader)

        train_risk_scores = np.array(train_risk_scores)
        train_survival_times = np.array(train_survival_times)
        train_censor = np.array(train_censor, dtype=bool)
        train_c_index = c_index(train_survival_times, train_censor, train_risk_scores)

        model.eval()
        val_loss = 0.0
        val_risk_scores = []
        val_survival_times = []
        val_censor = []

        with torch.no_grad():
            for inputs, survival_times, censor in val_loader:
                inputs = inputs.to(device)
                survival_times = survival_times.to(device)
                censor = censor.to(device)

                outputs = model(inputs)
                risk_scores = outputs.logits.squeeze()  # Remove the extra dimension
                loss = loss_fn(risk_scores, survival_times, censor)

                val_loss += loss.item()
                val_risk_scores.extend(risk_scores.detach().cpu().numpy())
                val_survival_times.extend(survival_times.detach().cpu().numpy())
                val_censor.extend(censor.detach().cpu().numpy())

        val_loss /= len(val_loader)

        val_risk_scores = np.array(val_risk_scores)
        val_survival_times = np.array(val_survival_times)
        val_censor = np.array(val_censor, dtype=bool)
        val_c_index = c_index(val_survival_times, val_censor, val_risk_scores)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train C-index: {train_c_index:.3f}, "
              f"Val Loss: {val_loss:.4f}, Val C-index: {val_c_index:.3f}")

        save_checkpoint(model, optimizer, epoch + 1)


def main():
    # Initialize the model with the pretrained checkpoint
    model = initialize_model(checkpoint_path, device)

    # Create the dataset
    dataset = SurvivalDataset(csv_file, photo_folder)

    # Split the dataset into training and validation sets
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Apply data augmentation to the training set
    train_dataset.dataset.transform = train_transform
    # Apply data augmentation to the validation set
    val_dataset.dataset = SurvivalDataset(csv_file, photo_folder, transform=val_transform)

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)

    # Create the optimizer
    optimizer = create_optimizer(model, base_lr=learning_rate, weight_decay=0.05, layer_decay=0.75, beta1=0.9, beta2=0.999)

    # Train the model
    train(model, train_loader, val_loader, optimizer, device, epochs, smooth_reg)

    print("Finetuning completed.")


if __name__ == "__main__":
    main()
