import math

import matplotlib.pyplot as plt
import torch as T

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class PhSigmoidLinear(T.nn.Module):

    def __init__(self, A1=0.060, A2=1.005, x0=0.145, d=0.033, cutoff=2):
        super(PhSigmoidLinear, self).__init__()
        self.A1 = A1
        self.A2 = A2
        self.x0 = x0
        self.d = d
        self.cutoff = cutoff

    def forward(self, x):
        x = x - self.x0
        x.clamp_(max=self.cutoff)
        return self.A2 + (self.A1 - self.A2) / (1 + T.exp(x / self.d))

    def derivative(self, x):
        return -((self.A1 - self.A2) * T.exp(x / self.d)) / (
            self.d * (T.exp(x / self.d) + 1)**2)

    def plot(self):
        x = T.linspace(-0.5, 1, 100).to(DEVICE)
        y = self.forward(x)

        y_min_i, y_min_val = T.argmin(y), T.min(y)
        y_max_i, y_max_val = T.argmax(y), T.max(y)
        print("Min g({:.2e})={:.2e}, Max g({:.2e})={:.2e}".format(
            x[y_min_i], y_min_val, x[y_max_i], y_max_val))

        fig, ax1 = plt.subplots(1, 1)
        ax1.set_title("Photonic Sigmoid")
        ax1.plot(x.cpu().numpy(),
                 y.cpu().numpy(),
                 color='b',
                 label='Activation')

        dy = self.derivative(x)

        dy_min_i, dy_min_val = T.argmin(dy), T.min(dy)
        dy_max_i, dy_max_val = T.argmax(dy), T.max(dy)
        print("Min g({:.2e})={:.2e}, Max g({:.2e})={:.2e}".format(
            x[dy_min_i], dy_min_val, x[dy_max_i], dy_max_val))

        ax1.plot(x.cpu().numpy(),
                 dy.cpu().numpy(),
                 color='r',
                 label='Derivative')
        plt.legend(loc='lower right')
        plt.show()

    def __repr__(self):
        return 'PhotonicSigmoid'


class PhSinusoidalLinear(T.nn.Module):

    def __init__(self, x_lower=0, x_upper=1):
        super(PhSinusoidalLinear, self).__init__()
        self.x_lower = x_lower
        self.x_upper = x_upper
        self.y_upper = 2

    def forward(self, x):
        x = x.clamp(self.x_lower, self.x_upper)
        return T.pow(T.sin(x * math.pi / 2.0), self.y_upper)

    def derivative(self, x):
        return math.pi * T.sin(math.pi * x / 2) * T.cos(math.pi * x / 2)

    def plot(self):
        x = T.linspace(self.x_lower, self.x_upper, 100).to(DEVICE)
        y = self.forward(x)

        y_min_i, y_min_val = T.argmin(y), T.min(y)
        y_max_i, y_max_val = T.argmax(y), T.max(y)
        print("Min g({:.2e})={:.2e}, Max g({:.2e})={:.2e}".format(
            x[y_min_i], y_min_val, x[y_max_i], y_max_val))

        fig, ax1 = plt.subplots(1, 1)
        ax1.set_title("Photonic Sinusoidal")
        ax1.plot(x.cpu().numpy(),
                 y.cpu().numpy(),
                 color='b',
                 label='Activation')

        dy = self.derivative(x)

        dy_min_i, dy_min_val = T.argmin(dy), T.min(dy)
        dy_max_i, dy_max_val = T.argmax(dy), T.max(dy)
        print("Min g({:.2e})={:.2e}, Max g({:.2e})={:.2e}".format(
            x[dy_min_i], dy_min_val, x[dy_max_i], dy_max_val))

        ax1.plot(x.cpu().numpy(),
                 dy.cpu().numpy(),
                 color='r',
                 label='Derivative')
        plt.legend(loc='upper right')
        plt.show()

    def __repr__(self):
        return 'PhotonicSinusoidal'
