import torch
import torchvision.models as models
from tqdm import tqdm
import torch.nn.functional as F
import utils
import argparse
import os

parser = argparse.ArgumentParser(prog="Neural Style Transfer")
parser.add_argument("content", help="Path to the content reference image")
parser.add_argument("style", help="Path to the style reference image")
parser.add_argument(
    "--output_folder",
    help="Folder to save generated images. Default: Folder named 'generated' in cwd.",
    default=os.path.join(os.getcwd(), "generated"),
)
parser.add_argument(
    "--gif",
    help="If argument specified, additionally create gif from generated images",
    action="store_true",
)
parser.add_argument(
    "--alpha",
    help="Ratio of style to content when computing loss. Use larger value to capture more style. Default: 10000. Typical value: between 100 and 100000",
    default=10000,
    type=int,
)
parser.add_argument(
    "--epochs", help="Number of iterations. Default: 300", type=int, default=300
)
parser.add_argument(
    "--img_size",
    help="Width of output image. Aspect ratio is set to 4:3. Default: 640",
    type=int,
    default=640,
)
parser.add_argument(
    "--device",
    help="'cuda' for GPU, 'cpu' for CPU. By default, selects GPU if available",
)
parser.add_argument(
    "--log_interval",
    help="Number of iterations after which to save image. Default: 50",
    type=int,
    default=50,
)
args = parser.parse_args()

img_x = args.img_size
img_y = int(3 * img_x / 4)
if args.device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = args.device

print(f"\nDevice: {device}")
print(f"Alpha: {args.alpha}")
print(f"Output folder: {args.output_folder}")

# C is content image, S is style image
print("\nLoading reference images...")
C, S = utils.load_references(args.content, args.style, img_y, img_x)
C, S = C.to(device), S.to(device)

# load model and disable gradients for the model
print("Loading VGG16 model...")
model = models.vgg16(pretrained=True).to(device)
for param in model.parameters():
    param.requires_grad = False

# G is generated image. Initialize it as the content image, with gradients enabled
G = torch.tensor(C.cpu().numpy(), requires_grad=True, device=device)

# LBFGS works best, better than Adam
optimizer = torch.optim.LBFGS([G])

# indices corresponding to the layers of VGG16
model_layers = [(1, 8), (8, 15), (15, 22), (22, 29)]

losses = {"content": [], "style": [], "total": []}

# training loop
pbar = tqdm(total=args.epochs + 20, leave=True, desc="Iterating...")
epoch = 0
while epoch <= args.epochs:

    # function for the optimizer to call
    def engine():
        global epoch

        # compute content loss
        content_G = model.features[:20](G)
        content_C = model.features[:20](C)
        loss_content = F.mse_loss(content_G, content_C)

        # get intermediate representations for style loss
        layers_G = [model.features[0](G)]
        layers_S = [model.features[0](S)]
        for idx in model_layers:
            layers_G.append(model.features[idx[0] : idx[1]](layers_G[-1]))
            layers_S.append(model.features[idx[0] : idx[1]](layers_S[-1]))

        # compute style loss using gram matrix
        style_losses = []
        for g, s in zip(layers_G, layers_S):
            num_channels = g.shape[1]
            num_pixels = g.shape[2]
            factor = 4 * num_channels * num_channels * num_pixels * num_pixels
            style_losses.append(
                F.mse_loss(utils.gram_matrix(g), utils.gram_matrix(s)) / factor
            )

        loss_style = sum(style_losses) / len(
            style_losses
        )  # equal weights for each layer

        loss = loss_content + args.alpha * loss_style

        optimizer.zero_grad()
        loss.backward()

        # log metrics
        losses["content"].append(loss_content.item())
        losses["style"].append(args.alpha * loss_style.item())
        losses["total"].append(loss.item())
        if (epoch + 1) % args.log_interval == 0:
            utils.save_image(G.cpu().detach(), args.output_folder, epoch)
        pbar.update()
        epoch += 1

        return loss

    optimizer.step(engine)

pbar.close()
if args.gif:
    print("Creating GIF in output folder...")
    utils.save_gif(args.output_folder)
print("Saving loss log to output folder...")
utils.plot_logs(losses, args.output_folder)
