import cntk


class MnistModel():
    def __init__(self, input_dim, num_output_classes):
        self.model = None
        self.label = cntk.input_variable(num_output_classes)
        self.input = cntk.input_variable(input_dim)
        self.number_features = input_dim
        self.number_labels = num_output_classes
        self.__apply_input_transform()

    def get_loss(self):
        return cntk.cross_entropy_with_softmax(self.model, self.label)

    def get_classification_error(self):
        return cntk.classification_error(self.model, self.label)

    def softmax(self):
        return cntk.softmax(self.model)

    def __apply_input_transform(self):
        self.input_transform = self.input / 255
