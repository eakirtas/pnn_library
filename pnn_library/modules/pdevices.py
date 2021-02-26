import numpy as np
import torch
import torch.nn.functional as F


def er_module(er, data):
    mx = torch.max(data, axis=0)
    er = 10**(er / 10)  # TODO: modulator() works only for data >=0
    data += 1
    data /= mx.values
    data += 2 / er
    data /= (1 + 2 / er)
    data *= mx.values
    data -= 1

    return data


def fir_module(data, filt):
    def apply_fir(sig, filt):
        n = len(sig)
        k = len(filt)
        rcv = np.concatenate((sig, sig))[n - k:2 * n]
        y = np.convolve(rcv, filt)
        y = np.roll(y, -2 * (len(filt) - 1))[:n]
        return y

    mx = torch.max(data, axis=0)
    mn = torch.min(data, axis=0)
    mxv = np.array(mx.values.cpu().detach().numpy())
    mnv = np.array(mn.values.cpu().detach().numpy())
    for i in range(len(mxv)):
        # if not mxv[i] == mnv[i]:
        data[:, i] = torch.from_numpy(
            np.float32(apply_fir(data[:, i].cpu().detach().numpy(), filt)))

    return data
