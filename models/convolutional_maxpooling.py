from .mnist_model import MnistModel
import cntk
from utils.activationfunctions.activations import swish;


class ConvolutionalMaxPooling(MnistModel):

    def __init__(self, input, output):
        super().__init__(input, output)
        self.create_model()

    def create_model(self):
        with cntk.layers.default_options(init=cntk.glorot_uniform(), activation=swish):
            h = self.input_transform
            h = cntk.layers.Convolution2D(filter_shape=(5, 5),
                                          num_filters=8,
                                          strides=(1, 1),
                                          pad=True, name="first_conv")(h)
            h = cntk.layers.MaxPooling(filter_shape=(2, 2),
                                       strides=(2, 2), name="first_max")(h)
            h = cntk.layers.Convolution2D(filter_shape=(5, 5),
                                          num_filters=16,
                                          strides=(1, 1),
                                          pad=True, name="second_conv")(h)
            h = cntk.layers.MaxPooling(filter_shape=(3, 3),
                                       strides=(3, 3), name="second_max")(h)
            self.model = cntk.layers.Dense(self.number_labels,
                                           activation=None, name="classify")(h)
