import torch
from torch import nn, optim
from torch.autograd import Variable


class LinearRegression(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        y = self.linear(x)
        return y


class LinearRegressionModel():
    def __init__(self, config):
        input_size = config.get('input_size', 1)
        output_size = config.get('output_size', 1)
        lr = config.get('lr', 0.001)
        momentum = config.get('momentum', 0.01)

        self.loss = nn.MSELoss()
        self.model = LinearRegression(input_size, output_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

    def _torch_var(self, value):
        return Variable(torch.Tensor(value))

    def train(self, epochs, x, y):
        epoch_loss = 0
        for epoch in range(epochs):
            y_pred = self.model(self._torch_var(x))
            loss = self.loss(y_pred, self._torch_var(y))
            epoch_loss = loss.data[0]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print(f'Epoch {epoch} loss: {epoch_loss}')

    def eval(self, x):
        y_pred = self.model(self._torch_var(x))
        return y_pred

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))