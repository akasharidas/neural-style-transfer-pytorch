import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations
from albumentations.pytorch.transforms import ToTensor

try:
    plt.style.use(
        "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
    )
except:
    pass


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def gram_matrix(filter):
    batch_size, nc, h, w = filter.shape
    g = filter.view(batch_size * nc, h * w)
    gram = torch.mm(g, g.t())
    return gram


def save_image(G, epoch):
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = unorm(G.squeeze())
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)
    plt.imsave(f"generated/generated_{epoch}.jpg", image)
    pass


def plot_logs(losses):
    fig, ax = plt.figure(figsize=(16, 9))
    sns.lineplot(x=list(range(len(losses["content"]))), y=losses["content"], ax=ax)
    sns.lineplot(x=list(range(len(losses["style"]))), y=losses["style"], ax=ax)
    sns.lineplot(x=list(range(len(losses["total"]))), y=losses["total"], ax=ax)
    ax.set_legend(["Content", "Style", "Total"])
    ax.set(xlabel="Epoch", ylabel="Loss", ylim=(0, 1000))
    fig.show()


def load_references(content, style, h, w):
    C_orig = plt.imread(content)
    S_orig = plt.imread(style)

    transform = albumentations.Compose(
        [
            albumentations.Resize(h, w, always_apply=True),
            albumentations.Normalize(always_apply=True),
            ToTensor(),
        ]
    )

    C = transform(image=C_orig)["image"].unsqueeze(0)
    S = transform(image=S_orig)["image"].unsqueeze(0)

    return C, S

