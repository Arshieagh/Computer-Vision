import torch
import matplotlib.pyplot as plt
from data.cifar100 import load_data
from models.baseline import BaseLineModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BaseLineModel().to(device)
model.load_state_dict(torch.load("model_final.pth", map_location=device))
model.eval()

activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


# Register hook on the first convolutional layer
model.stem[0].register_forward_hook(get_activation("conv1"))

# Load a single test image
_, test_loader = load_data(batch_size=1)
image, _ = next(iter(test_loader))
image = image.to(device)

# Forward pass (no gradients needed)
with torch.no_grad():
    model(image)

# Retrieve and visualise feature maps
feature_maps = activations["conv1"][0].cpu()  # shape: (C, H, W)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(feature_maps[i], cmap="viridis")
    ax.axis("off")

plt.suptitle("First Convolutional Layer Feature Maps")
plt.tight_layout()
plt.show()
