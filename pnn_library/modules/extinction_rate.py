import torch as T


class ExtinctionRate(T.nn.Module):
    def __init__(self, er_value):
        super().__init__()
        self.er_value = 10**(er_value / 10.0)

    def forward(self, x):
        x = x + 1.0
        x_max = T.max(x, axis=0).values

        x = x / x_max
        x = x + 2.0 / self.er_value
        x = x / (1.0 + 2.0 / self.er_value)
        x = x * x_max
        x = x - 1.0

        return x
