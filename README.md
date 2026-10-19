# FAHR-Face

Code accompanying **Foundation artificial intelligence models for health recognition by use of face photographs (FAHR-Face): a model development and validation study**, Haugg, Lee, and colleagues, *The Lancet Digital Health* (2026). [Paper](https://doi.org/10.1016/j.landig.2026.101056) · [Model checkpoints and loading examples](https://huggingface.co/fridolin22/FAHR-Face)

| Model | Description | Released checkpoint |
| --- | --- | --- |
| FAHR-Face | ViT masked autoencoder pretrained on more than 40 million WebFace42M photographs. | [fahr_face.pth](https://huggingface.co/fridolin22/FAHR-Face/resolve/main/fahr_face.pth?download=true) |
| FAHR-FaceAge | Age estimation model, fine-tuned on ten public datasets. | [fahr_faceage.pth](https://huggingface.co/fridolin22/FAHR-Face/resolve/main/fahr_faceage.pth?download=true) |

Released weights include **FAHR-Face and FAHR-FaceAge only**. FAHR-FaceSurvival training code is included in this repository; its weights are not released.

## Prepare images first

**FAHR-FaceAge requires cropped and aligned face images.** For raw photographs:

1. Detect, crop, and align one face per image using [other/face_crop_align.py](other/face_crop_align.py). Set its `input_dir` and `output_dir`, then run `python other/face_crop_align.py` in the project environment.
2. Use the resulting face crops for inference. The age inference example resizes them to **112 × 112 RGB pixels** and applies the required normalization.

Already cropped and aligned images can be used directly.

## Environment setup

The full training and face-preprocessing environment is provided in `facefoundation.yml`:

```bash
conda env create -f facefoundation.yml
conda activate facefoundation
```

For standalone age inference, the [model card](huggingface/README.md#setup) provides a smaller, tested Python 3.11 setup.

## Download and run

Both checkpoints are publicly available together at [fridolin22/FAHR-Face](https://huggingface.co/fridolin22/FAHR-Face). No Hugging Face login is required to download them.

Using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
hf download fridolin22/FAHR-Face fahr_face.pth fahr_faceage.pth --local-dir checkpoints
```

For **age inference**, follow the [standalone example](https://huggingface.co/fridolin22/FAHR-Face#fahr-faceage-inference), also maintained in [huggingface/README.md](huggingface/README.md#fahr-faceage-inference). Only `fahr_faceage.pth` is needed. Both released files are original training checkpoints containing `model_state_dict` and optimizer state.

The existing `run.py` combines age and survival inference. It additionally requires `checkpoints/fahr_facesurvival.pth`, which is not distributed. For users with that checkpoint, configure the image paths in `run.py` and run `python run.py`; set `perform_cropping_alignment=False` for images that are already cropped and aligned. Results are saved to `predicted_results.csv`.

## Training

Set the dataset and checkpoint paths in each script before training. Run commands from the indicated directory:

**Foundation model:**

```bash
cd training/training_fahr_face
python run.py
```

**Age model:**

```bash
cd training/train_fahr_faceage
python run_pretraining.py
python run_finetuning.py
```

**Survival model:**

```bash
cd training/train_fahr_facesurvival
python run.py
```

## Data and test images

WebFace42M is the cleaned subset of WebFace260M; access is managed by the original dataset authors. FAHR-FaceAge development used 743,152 photographs from ten public datasets; APPA-REAL (6,783 photographs) was used for evaluation, with a reported mean absolute error of 5.1 years. Clinical data are not distributed.

[Synthetic test images](https://www.dropbox.com/scl/fo/hm9a5x60vya2810gyqty9/AD4yp_v6UY-1HLqMKy902X8?rlkey=d8hfle133ecsvh4bp7ripzeq8&st=q9g357io&dl=0) are already cropped and aligned. Extract them to `synthetic_dataset_cropped_aligned/` and use them directly for age inference.

## Citation

Haugg F, Lee G, He J, et al. **Foundation artificial intelligence models for health recognition by use of face photographs (FAHR-Face): a model development and validation study.** *The Lancet Digital Health*. 2026. https://doi.org/10.1016/j.landig.2026.101056

Fridolin Haugg and Grace Lee contributed equally. [Earlier preprint](https://arxiv.org/abs/2506.14909).

## Ethical use

Intended for academic research. Use for discrimination, surveillance, or harmful applications is prohibited. Follow appropriate research ethics and obtain necessary approvals.

## Licensing

- **Code:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- **Model weights:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Associated article:** [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
