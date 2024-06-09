import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from nns.mlp import MLP

# Hyperparameters
epochs = 100
batch_size = 32
learning_rate = 1e-3
weight_decay = 1e-4
input_size = 28 * 28  # Image size of MNIST
hidden_size = 100
num_classes = 10

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

# Reduce dataset size and split into train/validation
train_dataset, _ = torch.utils.data.random_split(train_dataset, [20_000, 40_000])
val_dataset, _ = torch.utils.data.random_split(test_dataset, [1_000, 9_000])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Define MLP model
model = MLP(input_size, hidden_size, num_classes).to(device)

# Define optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# Define loss function
loss_func = nn.CrossEntropyLoss()

# Training loop
train_loss_all = []
val_loss_all = []
for epoch in range(epochs):
    # Train
    train_loss = 0
    train_num = 0
    model.train()
    with tqdm(train_loader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_num += images.size(0)

            pbar.set_postfix(lr=optimizer.param_groups[0]["lr"])
    train_loss_all.append(train_loss / train_num)
    pbar.set_postfix(loss=train_loss / train_num)

    # Validation
    model.eval()
    val_loss = 0
    val_num = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss += loss_func(outputs, labels).item() * images.size(0)
            val_num += images.size(0)
    val_loss_all.append(val_loss / val_num)

    # Update learning rate
    scheduler.step()

    print(f"Epoch {epoch + 1}, Val Loss: {val_loss / val_num}")

# Save the trained model
torch.save(model.state_dict(), "./model/mlp_mnist.pth")

# Plot the training curve
plt.figure(figsize=(10, 5))
plt.plot(train_loss_all, label="Train Loss")
plt.plot(val_loss_all, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Curve for MLP on MNIST")
plt.show()
