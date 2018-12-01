from .linear_regression import LinearRegressionModel
import os
from enum import Enum, auto
import numpy as np


# eventually dynamically pulled from db, retrain if new attribute arises
class Attributes(Enum):
    ZIPCODE = auto()
    ROOF_DAMAGE = auto()
    FLOOD_DAMAGE = auto()
    FOUNDATION_DAMAGE = auto()
    WATER_LEVEL = auto()
    INCOME = auto()
    DESTROYED = auto()
    HOUSEHOLD_SIZE = auto()
    HOME_TYPE = auto()

class HomeAssessor():
    def __init__(self, config):
        self.filepath = config.get('filepath', '')
        self.model_type = config.get('model', 'linear_regression')
        self.model_path = os.path.join(self.filepath, self.model_type)
        model_config = config[self.model_type]
        if self.model_type == 'linear_regression':
            model_config['input_size'] = len(Attributes)
            self.model = LinearRegressionModel(model_config)
        else:
            raise ValueError('model type unknown.')

    def _calculate_claim_amount(self, model_output):
        return model_output

    def _featurize_attributes(self, attributes):
        feature_array = np.zeros(len(Attributes))
        for key, value in attributes.items():
            index = Attributes[key].value
            feature_array[index] = value
        return feature_array

    def load(self):
        self.model.load(self.filepath)  # .pth

    def train(self, dataset):
        # repairAmount
        for input, label in dataset:
            # batch?
            self.model.train(input, label)
        self.model.save(self.filepath)

    def predict_from_attributes(self, attributes):
        features = self._featurize_attributes(attributes)
        y_pred = self.model.eval(features)
        claim_pred = self._calculate_claim_amount(y_pred)
        return claim_pred