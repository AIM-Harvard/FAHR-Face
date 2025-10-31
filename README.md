Foundation Artificial Intelligence Models for Health Recognition Using Face Photographs (FAHR-Face)

This repository contains the implementation of the FAHR (Foundation Artificial Intelligence Models for Health) models as described in the associated publication. The codebase includes three main components:

1. **FAHR-Face** - Foundation model trained on facial images using a Masked Autoencoder approach
2. **FAHR-FaceAge** - Age estimation model built on top of the foundation model
3. **FAHR-FaceSurvival** - Survival analysis model for health risk prediction

## Repository Structure

```
fahr_face_code/
├── run.py                        # Main script to run inference with trained models
├── ViTForAgeEstimation.py        # Age estimation model architecture
├── ViTForSurvivalAnalysis.py     # Survival analysis model architecture
├── facefoundation.yml            # Miniforge environment configuration
├── training/
│   ├── training_fahr_face/       # Training code for the foundation model
│   │   └── run.py                # Script to train the FAHR-Face foundation model
│   ├── train_fahr_faceage/       # Training code for the age estimation model
│   │   ├── run_pretraining.py    # Pretraining script for the age model
│   │   ├── run_finetuning.py     # Finetuning script for the age model
│   │   ├── loader.py             # Data loading utilities for age estimation
│   │   ├── optimizer.py          # Optimizer configuration
│   │   └── ViTForAgeEstimation.py # Age model architecture
│   └── train_fahr_facesurvival/  # Training code for the survival analysis model
│       ├── run.py                # Script to train the survival model
│       ├── concordance_index.py  # Evaluation metrics for survival analysis
│       ├── loader.py             # Data loading utilities for survival analysis
│       ├── optimizer.py          # Optimizer configuration
│       └── ViTForSurvivalAnalysis.py # Survival model architecture
└── other/
    └── face_crop_align.py        # Utilities for face cropping and alignment
```

## Model Architecture

### FAHR-Face (Foundation Model)

The foundation model is based on the Vision Transformer (ViT) architecture, specifically the Masked Autoencoder method. The model was trained on approximately 40 million high-quality facial images from the WebFace260M dataset.

Key features:
- Transformer-based architecture with self-attention mechanisms
- Trained using the Masked Autoencoder method where a high portion (75%) of the image patches are randomly masked
- The encoder processes the visible patches, and a learnable mask token is added at the positions of the masked patches
- The decoder reconstructs the raw pixel values for the masked positions based on the encoded visible patches

Several adjustments were made to adapt the "facebook/vit-mae-base" model for this use case:
- Image size reduced to 112x112 pixels to match the face photograph dataset
- Positional embeddings were resized for both the encoder and decoder
- Trained with a learning rate of 1.5e-5 and AdamW optimizer with a weight decay of 0.05

### FAHR-FaceAge (Age Estimation Model)

The age estimation model builds upon the foundation model to predict chronological age from facial images:
- Uses transfer learning from the pre-trained FAHR-Face model
- Fine-tuned on multiple age-labeled face datasets (IMDB-WIKI, KANFace, FGNET, CACD, AFAD, MegaAge, MORPH, LAG)
- Trained with L1 loss (Mean Absolute Error) to predict age as a continuous value
- Uses data augmentation techniques during training

### FAHR-FaceSurvival (Survival Analysis Model)

The survival analysis model is designed to predict health risks from facial images:
- Built on top of the FAHR-Face foundation model
- Trained using a ranking loss function optimized for survival analysis
- Evaluated using the concordance index (C-index) metric, which is only used for validation and not for training loss
- Incorporates smoothed regularization to improve model stability

## Usage

### Running Inference

To run inference with the trained models:

1. First, download the model checkpoints as described in the [Model Checkpoints](#model-checkpoints) section.

2. Download the test images as described in the [Test Images](#test-images) section.

3. Run the main script:
```bash
python run.py
```

The script is pre-configured with the following paths:
- `age_checkpoint_path`: "checkpoints/fahr_faceage.pth" - Path to the age estimation model
- `survival_checkpoint_path`: "checkpoints/fahr_facesurvival.pth" - Path to the survival analysis model
- `photo_folder`: "synthetic_dataset_cropped_aligned" - Path to the folder with test photos

You can modify these paths in the script if needed for your specific setup.

### Training Models

#### Foundation Model (FAHR-Face)

```bash
cd training/training_fahr_face
python run.py
```

#### Age Estimation Model (FAHR-FaceAge)

For pretraining:
```bash
cd training/train_fahr_faceage
python run_pretraining.py
```

For finetuning:
```bash
cd training/train_fahr_faceage
python run_finetuning.py
```

#### Survival Analysis Model (FAHR-FaceSurvival)

```bash
cd training/train_fahr_facesurvival
python run.py
```

## Dataset Information

The FAHR-Face foundation model was trained on the WebFace260M dataset, which contains over 40 million high-quality facial images spanning 4 million unique identities. The dataset features individuals from over 200 distinct countries/regions and more than 500 professions, providing broad diversity in nationality and background. Analysis of the dataset's characteristics shows an extensive range of poses (yaw angles from -90 to 90 degrees) and includes most major races worldwide, such as Caucasian, African, East Asian, South Asian, and more. The WebFace260M dataset covers a wide range of ages, with dates of birth going back to 1846, ensuring substantial age diversity in the face data.

The dataset's diversity extends to image quality and lighting conditions. The images were collected from different time periods and scenarios (both controlled and uncontrolled), so they vary in photo qualities and lighting situations. This variety in image characteristics helps train models that can generalize well to real-world applications where lighting and image quality may vary significantly.

## Environment Setup

This project requires Python 3.8 and uses Miniforge (a free and open-source conda distribution that uses conda-forge as the default channel). To set up the required environment:

1. Install Miniforge if you don't have it already:
   - Download from: https://github.com/conda-forge/miniforge

2. Create and activate the environment:
```bash
conda env create -f facefoundation.yml
conda activate facefoundation
```

## Model Checkpoints

We will release pretrained checkpoints upon publication. Until then, this repository contains code only. If you have access to internal weights, create a `checkpoints` directory in the repo root and place the files with the following names:
- `fahr_face.pth` — Foundation model
- `fahr_faceage.pth` — Age estimation model
- `fahr_facesurvival.pth` — Survival analysis model

## Test Images

The synthetic test images are available for download:

[Download Synthetic Test Images](https://www.dropbox.com/scl/fo/hm9a5x60vya2810gyqty9/AD4yp_v6UY-1HLqMKy902X8?rlkey=d8hfle133ecsvh4bp7ripzeq8&st=q9g357io&dl=0)

After downloading, extract the files to create a `synthetic_dataset_cropped_aligned` directory in the root of the repository. This folder contains pre-processed face images used for testing the models.

Alternatively, you can run the pipeline with your own images:
- For raw images, set `perform_cropping_alignment=True` in `run.py` to detect/align faces on the fly.
- For preprocessed images, set `perform_cropping_alignment=False` and point `photo_folder` to your cropped/aligned images (e.g., `synthetic_dataset_cropped_aligned`).

## Performance

### Inference Time
The FAHR models are designed to be computationally efficient. Inference time for a single image is:
- Less than 5 seconds on a standard consumer-grade CPU
- Even faster on GPU-enabled systems

## Citation

Please cite: Haugg et al., "Foundation Artificial Intelligence Models for Health Recognition Using Face Photographs (FAHR-Face)." arXiv:2506.14909 (2025). https://arxiv.org/abs/2506.14909

## Ethical Use

This software is intended for academic research. Any use for discrimination, surveillance, or harmful applications is prohibited. Users must follow appropriate research ethics and obtain necessary approvals.

## Licensing

This repository is multi-licensed:

- Code: Apache License 2.0.
- Model weights: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0). 

Full texts: https://www.apache.org/licenses/LICENSE-2.0 and https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
