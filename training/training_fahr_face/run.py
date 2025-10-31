import os
import math
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from transformers import ViTMAEForPreTraining, ViTMAEConfig, ViTImageProcessor
import numpy as np
from queue import Empty


class WebFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_processor):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.filepaths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.filepaths.append(os.path.join(subdir, file))
        self.filepaths = np.array(self.filepaths)  # Convert list to Numpy array
        print(f"Number of images found: {len(self.filepaths)}")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path)
        image = image.convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt")
        image.close()  # Explicitly close the image file
        return inputs['pixel_values'][0]


def resize_pos_embed(posemb, new_num_patches):
    cls_token_emb = posemb[:, 0:1, :]
    posemb_grid = posemb[:, 1:, :]
    old_h = old_w = int(math.sqrt(posemb_grid.size(1)))
    new_h = new_w = int(math.sqrt(new_num_patches))
    posemb_grid = posemb_grid.reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(new_h, new_w), mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).view(1, new_h * new_w, -1)
    return torch.cat([cls_token_emb, posemb_grid], dim=1)


def data_loader_worker(data_queue, dataset, batch_size):
    num_data = len(dataset)
    indices = list(range(num_data))
    np.random.shuffle(indices)

    for i in range(0, num_data, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch = [dataset[j] for j in batch_indices]
        data_queue.put(batch)
    data_queue.put(None)


def train(model, data_queue, device, optimizer):
    total_loss = 0
    while True:
        try:
            batch = data_queue.get(timeout=3)
            if batch is None:
                break

            batch = torch.stack([item for item in batch]).to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        except Empty:
            continue
    return total_loss


if __name__ == "__main__":
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    original_state_dict = original_model.state_dict()
    new_config = ViTMAEConfig.from_pretrained("facebook/vit-mae-base", image_size=112)
    new_model = ViTMAEForPreTraining(new_config)
    new_model.to(device)

    new_num_patches = (112 // new_config.patch_size) ** 2
    new_posemb_encoder = resize_pos_embed(original_state_dict['vit.embeddings.position_embeddings'], new_num_patches)
    original_state_dict['vit.embeddings.position_embeddings'] = new_posemb_encoder
    new_posemb_decoder = resize_pos_embed(original_state_dict['decoder.decoder_pos_embed'], new_num_patches)
    original_state_dict['decoder.decoder_pos_embed'] = new_posemb_decoder
    new_model.load_state_dict(original_state_dict)

    optimizer = torch.optim.AdamW(new_model.parameters(), lr=1.5e-5, weight_decay=0.05, betas=(0.9, 0.9999))

    # Initialize dataset
    image_processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base", size=112, do_resize=False,
                                                        do_normalize=True)
    dataset = WebFaceDataset("/path/to/WebFace260M", image_processor)

    # Set up multiprocessing
    mp.set_start_method('spawn')
    data_queue = mp.Queue(maxsize=10)
    data_process = mp.Process(target=data_loader_worker, args=(data_queue, dataset, 1024))
    data_process.start()

    # Training loop
    for epoch in range(100):
        epoch_loss = train(new_model, data_queue, device, optimizer)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

    data_process.join()

    # Save model checkpoint, etc.
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
    torch.save(
        {'epoch': epoch, 'model_state_dict': new_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
         'loss': epoch_loss}, checkpoint_path)
