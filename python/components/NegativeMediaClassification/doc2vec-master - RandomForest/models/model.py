from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self):
        self.model = None
        super().__init__()

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass
