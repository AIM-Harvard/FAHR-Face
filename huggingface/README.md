---
license: cc-by-nc-sa-4.0
library_name: pytorch
tags:
  - computer-vision
  - transformers
  - masked-autoencoder
  - age-estimation
  - facial-analysis
  - arxiv:2506.14909
---

# FAHR-Face

Checkpoints for FAHR-Face and FAHR-FaceAge, accompanying our *Lancet Digital Health* study. [Paper](https://doi.org/10.1016/j.landig.2026.101056) · [Code](https://github.com/AIM-Harvard/FAHR-Face)

| Model | Checkpoint | Purpose |
| --- | --- | --- |
| FAHR-Face | `fahr_face.pth` | Foundation model for facial representations and fine-tuning. |
| FAHR-FaceAge | `fahr_faceage.pth` | Facial age estimation in years. |

FAHR-Face is a ViT masked autoencoder pretrained on more than 40 million WebFace42M photographs. FAHR-FaceAge was fine-tuned on 743,152 photographs from ten public datasets and achieved a mean absolute error of 5.1 years on APPA-REAL (6,783 photographs). See the paper for methods and clinical evaluation.

FAHR-FaceSurvival weights are not publicly released. Both released files are PyTorch checkpoints containing `model_state_dict` and optimizer state.

## Prepare images first

**FAHR-FaceAge requires a cropped and aligned face image.** For raw photographs, detect, crop, and align one face first, then resize it to **112 × 112 RGB pixels**. The age example below assumes cropping and alignment are already complete and applies the required normalization.

Use the repository's [face cropping and alignment utility](https://github.com/AIM-Harvard/FAHR-Face/blob/main/other/face_crop_align.py), setting `input_dir` and `output_dir` before running it. It requires RetinaFace from the repository's preprocessing environment. Already cropped and aligned images can be used directly.

## Setup

Use Python 3.11 and run the examples from the code repository's root:

```bash
git clone https://github.com/AIM-Harvard/FAHR-Face.git
cd FAHR-Face
python -m pip install "torch==2.1.2" "transformers==4.38.2" "huggingface-hub==0.22.2" "numpy<2" pillow
```

### FAHR-Face foundation model

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import ViTMAEConfig, ViTMAEForPreTraining

checkpoint_path = hf_hub_download("fridolin22/FAHR-Face", "fahr_face.pth")
model = ViTMAEForPreTraining(ViTMAEConfig(image_size=112))
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()
```

For foundation-model preprocessing, follow [the training script](https://github.com/AIM-Harvard/FAHR-Face/blob/main/training/training_fahr_face/run.py), which uses the `facebook/vit-mae-base` image processor.

### FAHR-FaceAge inference

Only the age checkpoint is needed for age inference.

```python
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import ViTConfig
from ViTForAgeEstimation import ViTForAgeRegression

checkpoint_path = hf_hub_download("fridolin22/FAHR-Face", "fahr_faceage.pth")
model = ViTForAgeRegression(ViTConfig(image_size=112))
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

# Input must already contain one cropped and aligned face.
with Image.open("cropped_aligned_face.jpg") as image:
    image = image.convert("RGB").resize((112, 112), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
pixels = ((pixels - mean) / std).unsqueeze(0)
with torch.inference_mode():
    predicted_age = model(pixels).age_output.item()
print(f"Predicted facial age: {predicted_age:.2f} years")
```

The code repository's combined `run.py` also requires the unreleased survival checkpoint. Use the standalone age example above for this release.

## Research use

Intended for academic research. Image quality and demographic differences can affect performance; clinical findings come from retrospective cancer cohorts and require further validation before clinical use. Follow the code repository's ethical-use guidance. Training photographs and clinical data are not distributed here.

## Citation

Haugg F, Lee G, He J, et al. **Foundation artificial intelligence models for health recognition by use of face photographs (FAHR-Face): a model development and validation study.** *The Lancet Digital Health*. 2026. [doi:10.1016/j.landig.2026.101056](https://doi.org/10.1016/j.landig.2026.101056). Haugg and Lee contributed equally.

## License

**Weights:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). **Code:** [Apache 2.0](https://github.com/AIM-Harvard/FAHR-Face#licensing). **Article:** [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
