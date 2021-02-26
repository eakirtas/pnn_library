import torch as T
import torch.nn.functional as F

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class FirFilter(T.nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = T.tensor(kernel, dtype=T.float)
        self.kernel = T.flipud(self.kernel).view(1, 1, -1).to(DEVICE)
        # self.kernel.require_grad = False

    def forward(self, x):
        # n, m, k = x.size(0), x.size(1), (self.kernel.size(-1) - 1) // 2 + 1

        x = T.stack((x, x, x), dim=1).view(-1, x.size(1))

        n, m, k = x.size(0), x.size(1), (self.kernel.size(-1) - 1) // 2 + 1

        x = x.transpose(0, 1)
        x = T.cat((x[:, n - k:n], x, x[:, :k]), dim=1)

        x = x.view(m, 1, -1)

        x = F.conv1d(x, self.kernel)
        x = x.view(m, -1).transpose(0, 1)
        x = x[1:n:3]

        return x
