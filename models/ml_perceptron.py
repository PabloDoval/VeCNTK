from .mnist_model import MnistModel
import cntk


class MlPerceptron(MnistModel):

    NUM_HIDDEN_LAYERS = 2
    HIDDEN_LAYERS_DIM = 400

    def __init__(self, input, output):
        super().__init__(input, output)
        self.create_model()

    def create_model(self):
        with cntk.layers.default_options(init=cntk.layers.glorot_uniform(), activation=cntk.ops.relu):
            h = self.input_transform
            for _ in range(MlPerceptron.NUM_HIDDEN_LAYERS):
                h = cntk.layers.Dense(MlPerceptron.HIDDEN_LAYERS_DIM)(h)
            self.model = cntk.layers.Dense(
                self.number_labels, activation=None)(h)
