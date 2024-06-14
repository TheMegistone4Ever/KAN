from torch import relu, Tensor
from torch.nn import Linear

from nns.inn import INN


class MLP(INN):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP, self).__init__(input_size)
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)

    def forward(self, x: Tensor, update_grid=False):
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return x
