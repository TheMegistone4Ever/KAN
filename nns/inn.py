from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class INN(Module, ABC):
    def __init__(self, input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__input_size = input_size

    def get_input_size(self):
        return self.__input_size

    @abstractmethod
    def forward(self, x: Tensor, update_grid=False):
        pass
