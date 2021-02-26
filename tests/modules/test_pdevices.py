import numpy as np
import pytest
import torch as T
from numpy.testing import assert_almost_equal
from pnn_library.modules.extinction_rate import ExtinctionRate
from pnn_library.modules.fir_filter import FirFilter
from pnn_library.modules.pdevices import er_module, fir_module


class TestPDevices():
    def test_extinction_rate(self):
        my_data = T.randn((32, 154))
        org_data = my_data.clone()

        ext_rate_module = ExtinctionRate(er_value=24)

        my_output = ext_rate_module(my_data)
        org_output = er_module(24, org_data)

        assert_almost_equal(org_output.detach().numpy(),
                            my_output.detach().numpy(),
                            decimal=6)

    def test_fir_filter(self):
        actual_data = T.randn((100, 154))
        expected_data = actual_data.clone()

        my_fir_filter = FirFilter([0.05, 0.9, 0.05])

        actual_data = fir_module(actual_data, np.array([0.05, 0.9, 0.05]))
        expected_data = my_fir_filter(expected_data)

        print(T.mean(T.abs(actual_data - expected_data)**2))

        assert_almost_equal(actual_data.detach().numpy()[-15:],
                            expected_data.detach().numpy()[-15:],
                            decimal=1)
