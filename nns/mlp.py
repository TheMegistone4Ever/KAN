from torch import relu
from torch.nn import Module, Linear


class MLP(Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP, self).__init__()
        self.__input_size = input_size
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return x

    def get_input_size(self):  # TODO: move to interface
        return self.__input_size
