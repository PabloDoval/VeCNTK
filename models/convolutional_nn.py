from .mnist_model import MnistModel
import cntk


class ConvolutionalNN(MnistModel):

    def __init__(self, input, output):
        super().__init__(input, output)
        self.create_model()

    def create_model(self):
        with cntk.layers.default_options(init=cntk.glorot_uniform(), activation=cntk.relu):
            h = self.input_transform
            h = cntk.layers.Convolution2D(filter_shape=(5, 5),
                                          num_filters=8,
                                          strides=(2, 2),
                                          pad=True, name='first_conv')(h)
            h = cntk.layers.Convolution2D(filter_shape=(5, 5),
                                          num_filters=16,
                                          strides=(2, 2),
                                          pad=True, name='second_conv')(h)
            self.model = cntk.layers.Dense(
                self.number_labels, activation=None, name='classify')(h)
