from __future__ import annotations

from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, title, show
from torch import device
from torch import no_grad
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


def plot_training_curve(train_loss_all: list[float], val_loss_all: list[float]) -> type(None):
    """
    Plot the training curve

    :param train_loss_all:
    :param val_loss_all:
    :return:
    """

    figure(figsize=(10, 5))
    plot(train_loss_all, label="Train Loss")
    plot(val_loss_all, label="Val Loss")
    xlabel("Epoch")
    ylabel("Loss")
    legend()
    title("Training Curve for KAN on MNIST")
    show()


def define_hyperparameters() -> dict[str, int | float | str]:
    """
    Define the parameters for training

    :return: Parameters
    """

    return {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "input_size": 28 * 28,
        "hidden_size": 100,
        "num_classes": 10,
        "root": r".\data",
        "model_path": r".\model\kan_mnist.pth"
    }


def train_model(model, epochs, train_loader, val_loader, optimizer,
                loss_func=CrossEntropyLoss()) -> tuple[list[float], list[float]]:
    """
    Training loop with given number of epochs

    :param model: Torch model
    :param epochs: Epochs to loop through
    :param train_loader: Training data loader
    :param val_loader: Validation data loader
    :param optimizer: Optimizer for the model
    :param loss_func: Loss function
    :return: Train and validation loss
    """

    train_loss_all = []
    val_loss_all = []
    input_size = model.get_input_size()
    scheduler = ExponentialLR(optimizer, gamma=.8)
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
        with no_grad():
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

    return train_loss_all, val_loss_all
