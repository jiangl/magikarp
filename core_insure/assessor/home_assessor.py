from assessor.linear_regression import LinearRegressionModel
from assessor.simple_nn import NNModel

import os
from enum import Enum, auto
import numpy as np


# eventually dynamically pulled from db, retrain when new attribute arises
class Attributes(Enum):
    roofDamageAmount = auto()
    foundationDamageAmount = auto()
    floodDamageAmount = auto()
    waterLevel = auto()
    rentalAssistanceAmount = auto()
    rpfvl = auto()
    ppfvl = auto()


class HomeAssessor():
    def __init__(self, config):
        self.config = config
        self.filepath = config.get('filepath', '')
        self.model_type = config.get('model', 'linear_regression')
        self.model_path = os.path.join(self.filepath, self.model_type)
        model_config = config[self.model_type]
        model_config['input_size'] = len(Attributes)
        if self.model_type == 'linear_regression':
            self.model = LinearRegressionModel(model_config)
        elif self.model_type == 'simple_nn':
            self.model = NNModel(model_config)
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
        self.model.load(self.filepath)

    def save(self):
        self.model.save(self.filepath)

    def train(self, inputs, labels, input_val=None, label_val=None):
        self.model.train(inputs, labels, input_val, label_val)
        self.model.save(self.filepath)

    def predict_from_attributes(self, attributes):
        features = self._featurize_attributes(attributes)
        y_pred = self.model.eval(features)
        claim_pred = self._calculate_claim_amount(y_pred)
        return claim_pred