from typing import Tuple

from matplotlib.pyplot import subplots, suptitle, show
from torch import device, cuda, load, no_grad, argmax
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from nns import KAN, MLP
from utils import define_dimensions

model_path_mlp, model_path_kan, root = r".\model\mlp_mnist.pth", r".\model\kan_mnist.pth", r".\data"
input_width, input_height, batch_size = 28, 28, 32
input_size, hidden_size, num_classes = define_dimensions().values()

# Device configuration
device = device("cuda" if cuda.is_available() else "cpu")

# Load MNIST test dataset
test_dataset = MNIST(root=root, train=False, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Load trained models
model_mlp = MLP(input_size, hidden_size, num_classes).to(device)
model_mlp.load_state_dict(load(model_path_mlp))
model_mlp.eval()

model_kan = KAN([input_size, hidden_size, num_classes]).to(device)
model_kan.load_state_dict(load(model_path_kan))
model_kan.eval()


# Function to plot predictions
def plot_predictions(model: Module, data_loader: DataLoader, w: int = 5, h: int = 5,
                     figsize: Tuple[int, int] = (10, 15)) -> None:
    """
    Plot predictions made by the model on the test dataset.

    :param model: Model to evaluate
    :param data_loader: DataLoader for the test dataset
    :param w: width of the grid (default: 5)
    :param h: height of the grid (default: 5)
    :param figsize: Figure size (default: (10, 15))
    :return:
    """

    fig, axes = subplots(h, w, figsize=figsize)
    fig.tight_layout(pad=.5)

    with no_grad():
        for i in range(h * w):
            image, label = next(iter(data_loader))
            image = image[0].reshape(-1, input_size).to(device)  # Take the first image
            predicted_label = argmax(model(image)).item()

            # Get the label for the first image in the batch
            label = label[0].item()  # Extract the first label from the batch

            ax = axes[i // w, i % w]
            ax.imshow(image.cpu().reshape(input_width, input_height), cmap="gray")
            ax.axis("off")
            ax.set_title(f"{predicted_label}", color="g" if predicted_label == label else "r")

    suptitle(f"\"{model.__class__.__name__}\" Predictions")
    show()


if __name__ == "__main__":
    plot_predictions(model_mlp, test_loader)
    plot_predictions(model_kan, test_loader)
