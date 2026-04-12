"""Models package."""

from autoresearch.models.cnn import CNN
from autoresearch.models.ffnn import FFNN
from autoresearch.models.gru import GRU
from autoresearch.models.lstm import LSTM
from autoresearch.models.pc_enc_dec import PCEncDec
from autoresearch.models.pc_ffnn import PCFFNN
from autoresearch.models.pc_rnn import PCRNN
from autoresearch.models.resnet import ResNet18
from autoresearch.models.snn_baseline import SNNBaseline
from autoresearch.models.snn_cnn import SNNCNN
from autoresearch.models.snn_gru import SNNGRUModel
from autoresearch.models.snn_lstm import SNNLSTMModel
from autoresearch.models.snn_vanilla_rnn import SNNVanillaRNNModel
from autoresearch.models.vanilla_rnn import VanillaRNN

__all__ = [
    "FFNN",
    "CNN",
    "GRU",
    "LSTM",
    "PCEncDec",
    "PCFFNN",
    "PCRNN",
    "ResNet18",
    "SNNBaseline",
    "SNNCNN",
    "SNNGRUModel",
    "SNNLSTMModel",
    "SNNVanillaRNNModel",
    "VanillaRNN",
]
