import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from retinaface import RetinaFace
from ViTForAgeEstimation import ViTForAgeRegression
from ViTForSurvivalAnalysis import ViTForSurvivalAnalysis
from transformers import ViTConfig

# Define paths and parameters
age_checkpoint_path = "checkpoints/fahr_faceage.pth"
survival_checkpoint_path = "checkpoints/fahr_facesurvival.pth"
photo_folder = "synthetic_dataset_cropped_aligned"
output_csv = "predicted_results.csv"
perform_cropping_alignment = True  # Added variable to control cropping and alignment


# Function to load the age estimation model
def load_age_model(checkpoint_path, device):
    config = ViTConfig(image_size=112, num_hidden_layers=12, num_attention_heads=12)
    model = ViTForAgeRegression(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    return model


# Function to load the survival analysis model
def load_survival_model(checkpoint_path, device, num_labels):
    model = ViTForSurvivalAnalysis(
        ViTConfig(num_labels=num_labels, image_size=112, num_hidden_layers=12, num_attention_heads=12))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load age estimation model
age_model = load_age_model(age_checkpoint_path, device)

# Load survival analysis model
survival_model = load_survival_model(survival_checkpoint_path, device, num_labels=1)

# Prepare data transforms
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Get the list of photo filenames
photo_filenames = [f for f in os.listdir(photo_folder) if f.endswith(".jpg") or f.endswith(".png")]

# Initialize lists to store the results
photo_names = []
predicted_ages = []
risk_scores = []

# Process each photo
for photo_filename in photo_filenames:
    photo_path = os.path.join(photo_folder, photo_filename)

    if perform_cropping_alignment:
        # Detect faces in the image
        resp = RetinaFace.detect_faces(photo_path)

        # Check if exactly one face is detected
        if resp is not None and len(resp) == 1:
            faces = RetinaFace.extract_faces(img_path=photo_path, align=True)
            face_image = Image.fromarray(faces[0])
        else:
            print(f"Skipping {photo_filename} as no face or multiple faces were detected.")
            continue
    else:
        # Load the already cropped photo
        face_image = Image.open(photo_path)

    # Apply transforms to the face image
    face_tensor = transform(face_image)
    face_tensor = face_tensor.unsqueeze(0).to(device)

    # Predict age
    with torch.no_grad():
        age_output = age_model(face_tensor)
        predicted_age = age_output.age_output.item()

    # Predict risk score
    with torch.no_grad():
        survival_output = survival_model(face_tensor)
        risk_score = survival_output.logits.item()

    # Append the results to the lists
    photo_names.append(photo_filename)
    predicted_ages.append(predicted_age)
    risk_scores.append(risk_score)

# Create a DataFrame with the results
results_df = pd.DataFrame({
    "photo_name": photo_names,
    "predicted_age": predicted_ages,
    "risk_score": risk_scores
})

# Save the results to a CSV file
results_df.to_csv(output_csv, index=False)