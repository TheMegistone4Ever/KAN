import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from nns import KAN
from nns.mlp import MLP

# Hyperparameters
batch_size = 32
input_width = 28
input_height = 28
input_size = input_width * input_height
hidden_size = 100
num_classes = 10

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST test dataset
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Load trained models
model_mlp = MLP(input_size, hidden_size, num_classes).to(device)
model_mlp.load_state_dict(torch.load("./model/mlp_mnist.pth"))
model_mlp.eval()

model_kan = KAN([input_size, hidden_size, num_classes]).to(device)
model_kan.load_state_dict(torch.load("./model/kan_mnist.pth"))
model_kan.eval()


# Function to plot predictions
def plot_predictions(model, data_loader, model_name, w=5, h=5, figsize=(10, 15)):
    fig, axes = plt.subplots(h, w, figsize=figsize)
    fig.tight_layout(pad=.5)

    with torch.no_grad():
        for i in range(h * w):
            image, label = next(iter(data_loader))
            image = image[0].reshape(-1, input_size).to(device)  # Take the first image
            predicted_label = torch.argmax(model(image)).item()

            # Get the label for the first image in the batch
            label = label[0].item()  # Extract the first label from the batch

            ax = axes[i // w, i % w]
            ax.imshow(image.cpu().reshape(input_width, input_height), cmap="gray")
            ax.axis("off")
            ax.set_title(f"{predicted_label}", color="g" if predicted_label == label else "r")

    plt.suptitle(f"{model_name} Predictions")
    plt.show()


# Plot predictions for both models
plot_predictions(model_mlp, test_loader, "MLP")
plot_predictions(model_kan, test_loader, "KAN")
