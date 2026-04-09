from abc import ABC,abstractmethod


class BaseTrainer(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass