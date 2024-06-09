from torch import device, cuda, save
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from nns import KAN
from utils import plot_training_curve, define_hyperparameters, train_model

(epochs, batch_size, learning_rate, weight_decay, input_size, hidden_size, num_classes, root,
 model_path) = define_hyperparameters()

# Device configuration
device = device("cuda" if cuda.is_available() else "cpu")

# Load MNIST dataset
train_dataset = MNIST(root=root, train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root=root, train=False, transform=ToTensor())

# Reduce dataset size and split into train/validation
train_dataset, _ = random_split(train_dataset, [20_000, 40_000])
val_dataset, _ = random_split(test_dataset, [1_000, 9_000])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Define KAN model
model = KAN([input_size, hidden_size, num_classes]).to(device)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_loss_all, val_loss_all = train_model(model, epochs, train_loader, val_loader, optimizer)

save(model.state_dict(), model_path)

plot_training_curve(train_loss_all, val_loss_all)
