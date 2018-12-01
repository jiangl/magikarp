from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass