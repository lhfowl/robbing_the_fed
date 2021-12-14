from collections import namedtuple
import torch

def plot_data(cfg, user_data, setup, scale=False, print_labels=False):
    """Plot user data to output. Probably best called from a jupyter notebook."""
    import matplotlib.pyplot as plt  # lazily import this here

    dm = torch.as_tensor(cfg.mean, **setup)[None, :, None, None]
    ds = torch.as_tensor(cfg.std, **setup)[None, :, None, None]

    data = user_data["data"].clone().detach()
    labels = user_data["labels"].clone().detach() if user_data["labels"] is not None else None
    classes = [] # If you want to get class labels, you need to fill this in. 
                 # e.g. for CIFAR-10, you want classes = ['Airplane', 'Automobile', ...]
    if labels is None:
        print_labels = False

    if scale:
        min_val, max_val = data.amin(dim=[2, 3], keepdim=True), data.amax(dim=[2, 3], keepdim=True)
        # print(f'min_val: {min_val} | max_val: {max_val}')
        data = (data - min_val) / (max_val - min_val)
    else:
        data.mul_(ds).add_(dm).clamp_(0, 1)
    data = data.to(dtype=torch.float32)

    if data.shape[0] == 1:
        plt.axis("off")
        plt.imshow(data[0].permute(1, 2, 0).cpu())
        if print_labels:
            plt.title(f"Data with label {classes[labels]}")
    else:
        grid_shape = int(torch.as_tensor(data.shape[0]).sqrt().ceil())
        s = 24 if data.shape[3] > 150 else 6
        fig, axes = plt.subplots(grid_shape, grid_shape, figsize=(s, s))
        label_classes = []
        for i, (im, axis) in enumerate(zip(data, axes.flatten())):
            axis.imshow(im.permute(1, 2, 0).cpu())
            if labels is not None and print_labels:
                label_classes.append(classes[labels[i]])
            axis.axis("off")
        if print_labels:
            print(label_classes)

            
class data_cfg_default:
    size = (1_281_167,)
    classes = 1000
    shape = (3, 224, 224)
    normalize = True
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


class attack_cfg_default:
    type = "analytic"
    attack_type = "imprint-readout"
    label_strategy = "random"  # Labels are not actually required for this attack
    normalize_gradients = False
    impl = namedtuple("impl", ["dtype", "mixed_precision", "JIT"])("float", False, "")


