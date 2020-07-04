import torch
import torchvision.models as models
from tqdm import tqdm
import torch.nn.functional as F
import utils


EPOCHS = 500
log_interval = 20
alpha = 1000  # set between 1000 and 100000
img_x = 768
img_y = int(3 * img_x / 4)
device = "cuda" if torch.cuda.is_available() else "cpu"
content_reference = "heidelberg.jpg"
style_reference = "composition7.jpg"

# C is content image, S is style image
C, S = utils.load_references(content_reference, style_reference, img_y, img_x)
C, S = C.to(device), S.to(device)

# load model and disable gradients for the model
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
pbar = tqdm(total=EPOCHS)
epoch = 0
while epoch <= EPOCHS:

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

        loss = loss_content + alpha * loss_style

        optimizer.zero_grad()
        loss.backward()

        # log metrics
        losses["content"].append(loss_content.item())
        losses["style"].append(alpha * loss_style.item())
        losses["total"].append(loss.item())
        if (epoch + 1) % log_interval == 0:
            tqdm.write(
                f"Epoch: {epoch}, Content Loss: {loss_content.item()}, Style Loss: {loss_style.item()}, Loss: {loss.item()}\n"
            )
            utils.save_image(G.cpu().detach(), epoch)
        pbar.update()
        epoch += 1

        return loss

    optimizer.step(engine)

pbar.close()
