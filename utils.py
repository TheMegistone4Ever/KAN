from __future__ import annotations

from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, title, show
from torch import device, cuda, save, no_grad
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
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
        "root": r".\data",
        "model_path": r".\model\kan_mnist.pth",
    }


def define_dimensions() -> dict[str, int | float | str]:
    """
    Define the parameters for training

    :return: Parameters
    """

    return {
        "input_size": 28 * 28,
        "hidden_size": 100,
        "num_classes": 10,
    }


def train_model(model, model_device, epochs, train_loader, val_loader, optimizer,
                loss_func=CrossEntropyLoss()) -> tuple[list[float], list[float]]:
    """
    Training loop with given number of epochs

    :param model: Torch model
    :param model_device: Device to run the model on
    :param epochs: Epochs to loop through
    :param train_loader: Training data loader
    :param val_loader: Validation data loader
    :param optimizer: Optimizer for the model
    :param loss_func: Loss function
    :return: Train and validation loss
    """

    train_loss_all, val_loss_all = list(), list()
    input_size = model.get_input_size()
    scheduler = ExponentialLR(optimizer, gamma=.8)

    # Training
    for epoch in range(epochs):
        train_loss_sum, train_num = 0, 0
        model.train()
        with tqdm(train_loader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                loss = do_backpropagation(model, model_device, optimizer, loss_func, images, labels.to(model_device),
                                          input_size)
                train_loss_sum += loss * images.size(0)
                train_num += images.size(0)
                pbar.set_postfix(lr=optimizer.param_groups[0]["lr"])

        train_loss = train_loss_sum / train_num if train_num != 0 else 0
        train_loss_all.append(train_loss)

        # Validation
        model.eval()
        val_loss = calculate_val_loss(model, model_device, val_loader, loss_func)
        val_loss_all.append(val_loss)

        update_lr(scheduler)

        pbar.set_postfix(loss=train_loss)
        print(f"\nEpoch {epoch + 1}, Val Loss: {val_loss:,.4f}")

    return train_loss_all, val_loss_all


def calculate_val_loss(model, model_device, val_loader, loss_func=CrossEntropyLoss()) -> float:
    """
    Calculate the validation loss

    :param model: Torch model
    :param model_device: Device to run the model on
    :param val_loader: Validation data loader
    :param loss_func: Loss function
    :return: Validation loss
    """

    val_loss = 0
    val_num = 0
    input_size = model.get_input_size()
    with no_grad():
        for images, labels in val_loader:
            images = images.reshape(-1, input_size).to(model_device)
            val_loss += loss_func(model(images), labels.to(model_device)).item() * images.size(0)
            val_num += images.size(0)

    return val_loss / val_num


def do_backpropagation(model, model_device, optimizer, loss_func, images, labels, input_size) -> float:
    """
    Perform backpropagation

    :param model: Torch model
    :param model_device: Device to run the model on
    :param optimizer: Optimizer for the model
    :param loss_func: Loss function for the model
    :param images: Input images for the model
    :param labels: Labels for the images
    :param input_size: Input size for the model
    :return: Loss value calculated during backpropagation
    """

    images = images.reshape(-1, input_size).to(model_device)
    optimizer.zero_grad()
    loss = loss_func(model(images), labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def update_lr(scheduler) -> type(None):
    """
    Update the learning rate

    :param scheduler: Scheduler for the optimizer
    :return:
    """

    scheduler.step()


def evaluate_model(model):
    """
    Train the model

    :param model: Model to train
    :return:
    """

    epochs, batch_size, learning_rate, weight_decay, root, model_path = define_hyperparameters().values()

    # Device configuration
    model_device = device("cuda" if cuda.is_available() else "cpu")

    # Load MNIST dataset
    train_dataset = MNIST(root=root, train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(root=root, train=False, transform=ToTensor())

    # Reduce dataset size and split into train/validation
    train_dataset, _ = random_split(train_dataset, [20_000, 40_000])
    val_dataset, _ = random_split(test_dataset, [1_000, 9_000])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Define MLP model
    model = model.to(model_device)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loss_all, val_loss_all = train_model(model, model_device, epochs, train_loader, val_loader, optimizer)

    save(model.state_dict(), model_path)

    plot_training_curve(train_loss_all, val_loss_all)
