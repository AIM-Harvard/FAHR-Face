import torch.optim as optim
# moved from training/train_fahr_face_survivel to training/train_fahr_facesurvival

def param_groups_lrd(model, base_lr, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75):
    param_group_names = {}
    param_groups = {}

    # Determine the number of layers in the Vision Transformer model
    num_layers = len(model.vit.encoder.layer) + 1
    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim == 1 or name in no_weight_decay_list:
            group_decay = "no_decay"
            decay = 0.0
        else:
            group_decay = "decay"
            decay = weight_decay

        layer_id = get_layer_id_for_vit(name, num_layers)
        group_name = f"layer_{layer_id}_{group_decay}"

        if group_name not in param_group_names:
            scale = layer_scales[layer_id]
            initial_lr = base_lr * scale
            #print(f"Setting initial_lr for {group_name}: {initial_lr}")  # Added print statement
            param_group_names[group_name] = {"lr_scale": scale, "weight_decay": decay, "params": [], "initial_lr": initial_lr}
            param_groups[group_name] = {"lr_scale": scale, "weight_decay": decay, "params": [], "initial_lr": initial_lr, "lr": initial_lr}

        param_group_names[group_name]["params"].append(name)
        param_groups[group_name]["params"].append(param)



    return list(param_groups.values())



def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id for the Vision Transformer.
    """
    if 'vit.embeddings' in name or 'vit.patch_embeddings' in name or 'vit.cls_token' in name or 'vit.pos_embed' in name:
        return 0
    elif 'vit.encoder.layer' in name:
        return int(name.split('.')[3]) + 1
    else:
        return num_layers

# Function to create optimizer
def create_optimizer(model, base_lr=1e-4, weight_decay=0.05, layer_decay=0.75, beta1=0.9, beta2=0.999):
    param_groups = param_groups_lrd(model, base_lr, weight_decay, no_weight_decay_list=[], layer_decay=layer_decay)
    return optim.AdamW(param_groups, lr=base_lr, betas=(beta1, beta2))
