# FAHR-Face

Code accompanying **Foundation artificial intelligence models for health recognition by use of face photographs (FAHR-Face): a model development and validation study**, Haugg, Lee, and colleagues, *The Lancet Digital Health* (2026). [Paper](https://doi.org/10.1016/j.landig.2026.101056) · [Model checkpoints](https://huggingface.co/fridolin22/FAHR-Face)

This repository contains the implementation of the FAHR (Foundation Artificial Intelligence Models for Health) models as described in the associated publication. The codebase includes three main components:

1. **FAHR-Face** - Foundation model trained on facial images using a Masked Autoencoder approach
2. **FAHR-FaceAge** - Age estimation model built on top of the foundation model
3. **FAHR-FaceSurvival** - Survival analysis model for health risk prediction

**Released weights: FAHR-Face and FAHR-FaceAge only.** Both are publicly available on [Hugging Face](https://huggingface.co/fridolin22/FAHR-Face); FAHR-FaceSurvival weights are not publicly released.

**FAHR-FaceAge requires cropped and aligned face images before inference.** See [Face Preprocessing](#face-preprocessing).

## Repository Structure

```
FAHR-Face/
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

The foundation model is based on the Vision Transformer (ViT) architecture, specifically the Masked Autoencoder method. The model was trained on more than 40 million facial images from WebFace42M, the cleaned subset of the WebFace260M benchmark.

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
- Two-stage fine-tuning on 743,152 age-labeled photographs from ten public datasets: first eight oversampled datasets (IMDB-WIKI, KANFace, FG-NET, CACD, AFAD, MegaAge, MORPH, LAGENDA), then age-balanced fine-tuning on UTK and AgeDB
- Trained with L1 loss (Mean Absolute Error) to predict age as a continuous value
- Uses data augmentation techniques during training

### FAHR-FaceSurvival (Survival Analysis Model)

The survival analysis model is designed to predict health risks from facial images:

- Built on top of the FAHR-Face foundation model
- Trained using a ranking loss function optimized for survival analysis
- Evaluated using the concordance index (C-index) metric, which is only used for validation and not for training loss
- Incorporates smoothed regularization to improve model stability

## Usage

### Face Preprocessing

**Detect, crop, and align one face per image before running FAHR-FaceAge.** Set `input_dir` and `output_dir` in [other/face_crop_align.py](other/face_crop_align.py), then run:

```bash
python other/face_crop_align.py
```

Use the resulting face crops as input. The age inference example converts them to 112 × 112 RGB pixels and applies the required normalization. Images that are already cropped and aligned can be used directly.

### Running Inference

For **standalone age inference with the released checkpoint**, use the [Hugging Face example](https://huggingface.co/fridolin22/FAHR-Face#fahr-faceage-inference), also maintained in [huggingface/README.md](huggingface/README.md#fahr-faceage-inference). It requires only `fahr_faceage.pth`.

The existing `run.py` combines age and survival inference. It also requires `checkpoints/fahr_facesurvival.pth`, which is not distributed. If you already have access to that checkpoint:

1. First, download the model checkpoints as described in the [Model Checkpoints](#model-checkpoints) section.

2. Download the test images as described in the [Test Images](#test-images) section.

3. Set `perform_cropping_alignment=False` when using images that are already cropped and aligned, then run the main script:

```bash
python run.py
```

The script is pre-configured with the following paths:

- `age_checkpoint_path`: "checkpoints/fahr_faceage.pth" - Path to the age estimation model
- `survival_checkpoint_path`: "checkpoints/fahr_facesurvival.pth" - Path to the survival analysis model
- `photo_folder`: "synthetic_dataset_cropped_aligned" - Path to the folder with test photos

You can modify these paths in the script if needed for your specific setup.

### Training Models

Set the dataset and checkpoint paths in each script before training. Run each command from the indicated directory, starting at the repository root.

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

FAHR-Face was pretrained on WebFace42M, the cleaned subset of WebFace260M, obtained from the original dataset authors under their data-use agreement. Researchers seeking access should contact the WebFace260M authors and follow their access procedures. The training images cover varied photographic conditions, poses, and facial appearances.

FAHR-FaceAge development used 743,152 photographs from ten public datasets. The first training stage used IMDB-WIKI, KANFace, FG-NET, CACD, AFAD, MegaAge, MORPH, and LAGENDA; the second stage used UTK and AgeDB. APPA-REAL (6,783 photographs) was used for evaluation only, with a reported mean absolute error of 5.1 years.

Clinical evaluation used the Harvard-RT cohort (38,183 patients) and independent Maastro-RT cohort (4,892 patients). Clinical photographs and patient-level data are not distributed in this repository.

## Environment Setup

The original training and face-preprocessing environment uses Python 3.8 and Miniforge (a free and open-source conda distribution that uses conda-forge as the default channel). For standalone age inference, the [model card](huggingface/README.md#setup) also provides a tested Python 3.11 setup. To create the original environment:

1. Install Miniforge if you don't have it already:
   - Download from: https://github.com/conda-forge/miniforge

2. Create and activate the environment:
```bash
conda env create -f facefoundation.yml
conda activate facefoundation
```

## Model Checkpoints

The released **FAHR-Face and FAHR-FaceAge** checkpoints are hosted together on [Hugging Face](https://huggingface.co/fridolin22/FAHR-Face). No login is required. Download them directly:

- [fahr_face.pth](https://huggingface.co/fridolin22/FAHR-Face/resolve/main/fahr_face.pth?download=true) — Foundation model
- [fahr_faceage.pth](https://huggingface.co/fridolin22/FAHR-Face/resolve/main/fahr_faceage.pth?download=true) — Age estimation model

Using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), download both into the `checkpoints` folder:

```bash
hf download fridolin22/FAHR-Face fahr_face.pth fahr_faceage.pth --local-dir checkpoints
```

These are original training checkpoints containing `model_state_dict` and optimizer state. Loading examples are available in the [model card](https://huggingface.co/fridolin22/FAHR-Face). FAHR-FaceAge inference needs only the age checkpoint. FAHR-FaceSurvival weights are not publicly released.

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

Please cite: Haugg F, Lee G, He J, et al. **Foundation artificial intelligence models for health recognition by use of face photographs (FAHR-Face): a model development and validation study.** *The Lancet Digital Health*. 2026. https://doi.org/10.1016/j.landig.2026.101056

Fridolin Haugg and Grace Lee contributed equally. The [earlier preprint](https://arxiv.org/abs/2506.14909) is also available.

## Ethical Use

This software is intended for academic research. Any use for discrimination, surveillance, or harmful applications is prohibited. Users must follow appropriate research ethics and obtain necessary approvals.

## Licensing

This repository is multi-licensed:

- Code: Apache License 2.0.
- Model weights: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).
- Associated journal article: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0).

Full texts: https://www.apache.org/licenses/LICENSE-2.0 and https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
