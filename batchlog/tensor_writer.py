from cntk.logging.progress_print import TensorBoardProgressWriter
from cntk.ops import reduce_mean
from settings import TENSOR_LOG_DIR


class TensorWriter():

    def __init__(self, model):
        self.__model = model
        self.__tensorboard_writer = TensorBoardProgressWriter(
            freq=20, log_dir=TENSOR_LOG_DIR, model=self.__model)

    def get_writer(self):
        return self.__tensorboard_writer

    def write_model_params(self, batch_id):
        for p in self.__model.parameters:
            self.__tensorboard_writer.write_value(
                "{}_{}/mean".format(p.uid, p.name),
                reduce_mean(p).eval(), batch_id)

    def write_image(self, image_data, count, name='trainning'):
        self.__tensorboard_writer.write_image(name, image_data, count)
