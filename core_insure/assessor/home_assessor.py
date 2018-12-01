from .LinearRegression import LinearRegressionModel

class HomeAssessor():
    def __init__(self, config):
        self.filepath = config.get('filepath', '')
        self.model_type = config.get('model', 'linear_regression')
        self.model = LinearRegressionModel(config[self.model_type])

    def load(self):
        self.model.load(self.filepath)  # .pth

    def train(self, dataset):
        for input, label in dataset:
            # batch?
            self.model.train(input, label)
        self.model.save(self.filepath)

