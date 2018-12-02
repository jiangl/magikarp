from abc import ABC, abstractmethod
class BaseDataLoader():
    def __init__(self, config):
        super().__init__()

    @abstractmethod
    def load_attributes(self, house_id):
        pass

    @abstractmethod
    def save_attributes(self, house_id, attribute_keys_values):
        pass

    @abstractmethod
    def update_claim(self, house_id, claim):
        pass

    @abstractmethod
    def load_houses(self, lat_long1, lat_long2):
        pass

    @abstractmethod
    def disconnect(self):
        pass