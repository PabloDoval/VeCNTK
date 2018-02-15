import cntk as C
from utils.activationfunctions.activations import swish

class Autoencoder():

    def __init__(self, input_dim, num_output_classes, transformation = None):
        self.__input_dim = input_dim.shape
        self.__features = input_dim
        self.__transformed_features = input_dim if transformation is None else transformation(input_dim)
        self.__num_output_classes = num_output_classes       
        self.encoded_model = None
        self.decoded_model = None        
        self.classifier = None

    def create_autoencoder(self):       
        with C.layers.default_options(init=C.layers.glorot_uniform()):

            conv_0 = C.layers.Convolution2D((10, 10), 10, strides=(2, 2), pad=True, activation=swish, name='conv_0')(self.__transformed_features)

            conv_1 = C.layers.Convolution2D((10, 10), 10, strides=(2, 2), pad=True, activation=swish, name='conv_1')(conv_0)

            conv_2 = C.layers.Convolution2D((10, 10), 1, strides=(2, 2), pad=True, activation=swish, name='conv_2')(conv_1)

            self.encoded_model = conv_2

            deconv_2 = C.layers.ConvolutionTranspose2D((10, 10), 10, strides=(2, 2), pad=True, name='deconv_2', output_shape=(conv_1.shape[1], conv_1.shape[2]))(self.encoded_model)

            deconv_1 = C.layers.ConvolutionTranspose2D((10, 10), 10, strides=(2, 2), pad=True, name='deconv_1', output_shape=(conv_0.shape[1], conv_0.shape[2]))(deconv_2)

            deconv_0 = C.layers.ConvolutionTranspose2D((10, 10), 1, strides=(2, 2), pad=True, name='deconv_0', output_shape=(self.__input_dim[1],self.__input_dim[2]))(deconv_1)

            self.decoded_model = deconv_0

            # Classifier
            dense_layer = C.layers.Dense(200, init=C.glorot_uniform(), activation=swish, name='dense_classifier')
            dense_output = C.layers.Dense(self.__num_output_classes, init=C.glorot_uniform(), activation=None, name='label_classifier')(dense_layer)
            output_shape = C.input_variable(self.encoded_model.shape)
            self.classifier = dense_output(output_shape)

            # import pdb
            # pdb.set_trace()

            return self.decoded_model