import torch
from torch import nn, optim
from torch.autograd import Variable
from assessor.base_model import BaseModel
import os


class LinearRegression(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        y = self.linear(x)
        return y


class LinearRegressionModel(BaseModel):
    def __init__(self, config):
        input_size = config.get('input_size', 1)
        output_size = config.get('output_size', 1)
        lr = config.get('lr', 0.001)
        momentum = config.get('momentum', 0)
        self.epochs = config.get('epochs', 100)

        self.loss = nn.SmoothL1Loss()
        self.model = LinearRegression(input_size, output_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)


    def _torch_var(self, value):
        return Variable(torch.Tensor(value))

    def train(self, x_inputs, y_labels, x_val=None, y_val=None):
        for epoch in range(self.epochs):
            y_pred = self.model(self._torch_var(x_inputs))
            loss = self.loss(y_pred, self._torch_var(y_labels))

            val_loss = 0
            if x_val and y_val:
                y_pred_val = self.model(self._torch_var(x_val))
                val_loss = self.loss(y_pred_val, self._torch_var(y_val))

            print(f'Epoch {epoch}, Train_Loss: {loss}, Val_Loss: {val_loss}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self, x):
        y_pred = self.model(self._torch_var(x))
        formatted_y = y_pred.data.numpy()
        return formatted_y

    def save(self, filepath):
        model_path = os.path.join(filepath, 'linear_regression.pth')
        torch.save(self.model.state_dict(), model_path)

    def load(self, filepath):
        model_path = os.path.join(filepath, 'linear_regression.pth')
        self.model.load_state_dict(torch.load(model_path))