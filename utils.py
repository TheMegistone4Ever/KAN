from __future__ import annotations

from typing import Tuple, List, Dict, Union, Any, Callable

from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, title, show
from torch import device, cuda, save, no_grad, Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from nns.inn import INN


def plot_training_curve(model: INN, train_loss_all: List[float], val_loss_all: List[float]) -> None:
    """
    Plot the training curve

    :param model: Model which results are plotted
    :param train_loss_all: Training losses
    :param val_loss_all: Validation losses
    :return:
    """

    figure(figsize=(10, 5))
    plot(train_loss_all, label="Train Loss")
    plot(val_loss_all, label="Val Loss")
    xlabel("Epoch")
    ylabel("Loss")
    legend()
    title(f"Training Curve for {model.__class__.__name__} on MNIST")
    show()


def define_hyperparameters() -> Dict[str, Union[int, float, str]]:
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


def define_dimensions() -> Dict[str, int]:
    """
    Define the parameters for training

    :return: Parameters
    """

    return {
        "input_size": 28 * 28,
        "hidden_size": 100,
        "num_classes": 10,
    }


def train_model(model: INN, model_device: Any, epochs: int, train_loader: DataLoader,
                val_loader: DataLoader, optimizer: Optimizer,
                loss_func: CrossEntropyLoss = CrossEntropyLoss()) -> Tuple[List[float], List[float]]:
    """
    Training loop with given number of epochs

    :param model: Torch model
    :param model_device: Device to run the model on
    :param epochs: Epochs to loop through
    :param train_loader: Training data loader
    :param val_loader: Validation data loader
    :param optimizer: Optimizer for the model
    :param loss_func: Loss function (default: CrossEntropyLoss)
    :return: Train and validation loss
    """

    train_loss_all, val_loss_all = list(), list()
    input_size = model.get_input_size()
    scheduler = ExponentialLR(optimizer, gamma=.8)
    for epoch in range(epochs):
        model.train()
        train_loss_all.append(run_step(f"Epoch {epoch + 1:05d}/{epochs}", train_loader, do_backpropagation,
                                       optimizer=optimizer, model=model, model_device=model_device, loss_func=loss_func,
                                       input_size=input_size))
        model.eval()
        val_loss_all.append(run_step("Validation Step", val_loader, calculate_val_loss, model=model,
                                     model_device=model_device, loss_func=loss_func, input_size=input_size))
        update_lr(scheduler)
    return train_loss_all, val_loss_all


def run_step(desc: str, data_loader: DataLoader, step_function: Callable, **kwargs: Any) -> float:
    """
    Run a step function on the data loader

    :param desc: Description for the progress bar
    :param data_loader: DataLoader to run the step function on
    :param step_function: Function to run on the data loader
    :param kwargs: Additional arguments for the step function
    :return: Average loss for the step function
    """

    loss_sum, num_samples, avg_loss = 0, 0, 0
    with tqdm(data_loader, desc=desc) as pbar:
        for images, labels in pbar:
            loss_sum += step_function(images=images, labels=labels, **kwargs) * images.size(0)
            num_samples += images.size(0)
            avg_loss = loss_sum / num_samples if num_samples else 0
            pbar.set_postfix_str((f"lr={kwargs['optimizer'].param_groups[0]['lr']:.2e}, "
                                  if "optimizer" in kwargs else "") + f"{avg_loss=:.4f}")
    return avg_loss


def calculate_val_loss(model: INN, model_device: Any, loss_func: Callable, images: Tensor, labels: Tensor,
                       input_size: int) -> float:
    """
    Calculate validation loss

    :param model: Torch model
    :param model_device: Device to run the model on
    :param images: Input images for the model
    :param labels: Labels which the model should predict
    :param loss_func: Loss function for the model
    :param input_size: Input size for the model
    :return: Loss value calculated during backpropagation
    """

    images = images.reshape(-1, input_size).to(model_device)
    labels = labels.to(model_device)
    with no_grad():
        outputs = model(images)
        loss = loss_func(outputs, labels)
    return loss.item()


def do_backpropagation(model: INN, model_device: Any, loss_func: Callable, images: Tensor, labels: Tensor,
                       input_size: int, optimizer: Optimizer) -> float:
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
    labels = labels.to(model_device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = loss_func(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def update_lr(scheduler: LRScheduler) -> None:
    """
    Update the learning rate

    :param scheduler: Scheduler for the optimizer
    :return:
    """

    scheduler.step()


def evaluate_model(model: INN) -> None:
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

    plot_training_curve(model, train_loss_all, val_loss_all)
