import torch as T


class SotonLayer(T.nn.Module):

    def __init__(self, in_features: int, out_features: int, has_bias=False):
        super().__init__()

        self.weight = T.nn.Parameter(T.randn((1, in_features)).cuda())
        self.has_bias = has_bias

        if has_bias:
            self.bias = T.nn.Parameter(
                T.full((out_features, ), 0, dtype=T.float).cuda())

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x * self.weight
        x = x.view(-1, x.size(1) // self.out_features, self.out_features)
        x = x.sum(dim=2)

        if self.has_bias:
            x = x + self.bias
        return x
