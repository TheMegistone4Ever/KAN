from matplotlib.pyplot import subplots, suptitle, show
from torch import device, cuda, load, no_grad, argmax
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from nns import KAN, MLP
from utils import define_hyperparameters

# Hyperparameters
model_path_mlp, model_path_kan = r".\model\mlp_mnist.pth", r".\model\kan_mnist.pth"
input_width, input_height = 28, 28
_, batch_size, _, _, input_size, hidden_size, num_classes, root, _ = define_hyperparameters()

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
def plot_predictions(model, data_loader, w=5, h=5, figsize=(10, 15)):
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

    suptitle(f"{model.__class__.__name__} Predictions")
    show()


# Plot predictions for both models
plot_predictions(model_mlp, test_loader)
plot_predictions(model_kan, test_loader)
