import torch as T

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class GaussianNoise(T.nn.Module):
    def __init__(self, std=None, normalized=False, is_active=True):
        super().__init__()
        self.std = std
        self.normalized = normalized
        self.is_active = is_active

    def forward(self, x, layer=None):
        if self.std is not None and self.std != 0.0 and self.is_active:
            noise = T.randn(x.size(), device=DEVICE) * self.std

            if self.normalized:
                noise *= T.std(x, axis=0, keepdim=True)

            x = x + noise.cuda()
        return x

    def activate(self):
        self.is_active = True

    def deactivate(self):
        self.is_active = False

    def set_active(self, is_active):
        self.is_active = is_active

    def add_to_weight(self, w):
        if self.std is not None and self.std != 0.0:
            with T.no_grad():
                noise = T.randn(list(w.shape)) * self.std
                w.add_(noise.to(DEVICE))


class SimulatedNoiseModule():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, layer=None):
        if layer is None:
            noise = (T.randn(list(x.shape)).to(DEVICE) + self.mean) * self.std
        else:
            noise = (T.randn(list(x.shape)).to(DEVICE) +
                     self.mean[layer]) * self.std[layer]
        x = x + noise * T.std(x)
        return x
