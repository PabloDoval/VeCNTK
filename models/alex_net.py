from .mnist_model import MnistModel
import cntk
from cntk.layers import Convolution2D, Activation, MaxPooling, Dense, Dropout, default_options, Sequential
from cntk.initializer import normal
from cntk.ops import relu, minus, constant


class AlexNet(MnistModel):

    def __init__(self, input, output):
        super().__init__(input, output)
        self.create_model()

    def __local_response_normalization(self, k, n, alpha, beta, name=''):
        x = cntk.placeholder(name='lrn_arg')
        x2 = cntk.square(x)
        x2s = cntk.reshape(x2, (1, cntk.InferredDimension), 0, 1)
        W = cntk.constant(alpha / (2 * n + 1), (1, 2 * n + 1, 1, 1), name='W')
        y = cntk.convolution(W, x2s)
        b = cntk.reshape(y, cntk.InferredDimension, 0, 2)
        den = cntk.exp(beta * cntk.log(k + b))
        apply_x = cntk.element_divide(x, den)
        return apply_x

    def create_model(self):

        mean_removed_features = minus(
            self.input, constant(114), name='mean_removed_input')

        with default_options(activation=None, pad=True, bias=True):
            self.model = Sequential([
                Convolution2D((11, 11), 96, init=normal(0.01),
                              pad=False, name='conv1'),
                Activation(activation=relu, name='relu1'),
                self.__local_response_normalization(
                    1.0, 2, 0.0001, 0.75, name='norm1'),
                MaxPooling((3, 3), (2, 2), name='pool1'),
                Convolution2D((5, 5), 192, init=normal(
                    0.01), init_bias=0.1, name='conv2'),
                Activation(activation=relu, name='relu2'),
                self.__local_response_normalization(
                    1.0, 2, 0.0001, 0.75, name='norm2'),
                MaxPooling((3, 3), (2, 2), name='pool2'),
                Convolution2D((3, 3), 384, init=normal(0.01), name='conv3'),
                Activation(activation=relu, name='relu3'),
                Convolution2D((3, 3), 384, init=normal(
                    0.01), init_bias=0.1, name='conv4'),
                Activation(activation=relu, name='relu4'),
                Convolution2D((3, 3), 256, init=normal(
                    0.01), init_bias=0.1, name='conv5'),
                Activation(activation=relu, name='relu5'),
                MaxPooling((3, 3), (2, 2), name='pool5'),
                Dense(4096, init=normal(0.005), init_bias=0.1, name='fc6'),
                Activation(activation=relu, name='relu6'),
                Dropout(0.5, name='drop6'),
                Dense(4096, init=normal(0.005), init_bias=0.1, name='fc7'),
                Activation(activation=relu, name='relu7'),
                Dropout(0.5, name='drop7'),
                Dense(self.number_labels, init=normal(0.01), name='fc8')
            ])(mean_removed_features)
