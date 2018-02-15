from .mnist_model import MnistModel
import cntk


class LogisticRegression(MnistModel):
    def __init__(self, input, output):
        super().__init__(input, output)
        self.create_model()

    def create_model(self):
        w = cntk.Parameter((self.number_features,  self.number_labels),
                           init=cntk.glorot_uniform(), name='W')
        b = cntk.Parameter((self.number_labels,), init=0, name='b')
        self.model = cntk.times(self.input_transform, w) + b
