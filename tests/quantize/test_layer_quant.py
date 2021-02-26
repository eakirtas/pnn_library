import torch as T
from data_utils.loaders.images.mnist.classic_mnist_loader import get_mnist
from pnn_library.quantize.v2 import LayerQuantWrap, Normalized
from train_utils.runners.multiclass_runner import MulticlassRunner

_DEFAULT_LAYER_D = {
    'input': {
        'is_quant': True
    },
    'weight': {
        'is_quant': True
    },
    'bias': {
        'is_quant': True
    },
    'output': {
        'is_quant': True
    },
}


class Model(T.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = LayerQuantWrap(
            T.nn.Conv2d(1, 1, 3, stride=2),
            is_quant=False,
            num_bits=3,
            stat_lambda=lambda: Normalized(alpha=1, beta=2),
            layer_dict=_DEFAULT_LAYER_D,
        )

        self.linear2 = LayerQuantWrap(
            T.nn.Linear(169, 10),
            is_quant=False,
            num_bits=3,
            stat_lambda=lambda: Normalized(alpha=1, beta=2),
            layer_dict=_DEFAULT_LAYER_D,
        )

    def forward(self, x):
        x = T.relu(self.conv1(x))
        x = T.flatten(x, start_dim=1)
        x = self.linear2(x)

        return x

    def set_q_properties(self, is_quant):
        self.conv1.set_quant_mode(is_quant)
        self.linear2.set_quant_mode(is_quant)


def train_model(model):
    train_dl, eval_dl = get_mnist(128)
    optimizer = T.optim.Adam(model.parameters())
    runner = MulticlassRunner(T.nn.CrossEntropyLoss(), )

    train_loss, train_acc = runner.fit(model,
                                       optimizer,
                                       train_dl,
                                       num_epochs=10,
                                       verbose=0)


class TestLayerQuant():

    def test_quant(self):
        model_orig = Model().cuda()
        model_eval = Model().cuda()

        model_orig.set_q_properties(True)
        model_eval.set_q_properties(True)

        train_model(model_orig)

        state_dict_orig = model_orig.state_dict()
        model_eval.load_state_dict(state_dict_orig)

        print('==================== Quant ====================')

        model_orig.conv1.quant_layer()
        model_eval.conv1.quant_layer()
        model_orig.linear2.quant_layer()
        model_eval.linear2.quant_layer()

        assert T.allclose(model_orig.conv1.layer.weight,
                          model_eval.conv1.layer.weight)

        assert T.allclose(model_orig.conv1.layer.bias,
                          model_eval.conv1.layer.bias)

        assert T.allclose(model_orig.linear2.layer.weight,
                          model_eval.linear2.layer.weight)

        assert T.allclose(model_orig.linear2.layer.bias,
                          model_eval.linear2.layer.bias)

        print('==================== DeQuant ====================')

        model_orig.conv1.dequant_layer()
        model_eval.conv1.dequant_layer()
        model_orig.linear2.dequant_layer()
        model_eval.linear2.dequant_layer()

        assert T.allclose(model_orig.conv1.layer.weight,
                          model_eval.conv1.layer.weight)

        assert T.allclose(model_orig.conv1.layer.bias,
                          model_eval.conv1.layer.bias)

        assert T.allclose(model_orig.linear2.layer.weight,
                          model_eval.linear2.layer.weight)

        assert T.allclose(model_orig.linear2.layer.bias,
                          model_eval.linear2.layer.bias)
