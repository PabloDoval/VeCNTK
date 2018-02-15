import matplotlib.pyplot as plt
from datasets.mnistdata.ctf_reader import init_reader
import numpy as np


def predict_minist_images(model, test_file, batch_size=25):
    input_dim = model.number_features
    out = model.softmax()
    size = input_dim[1] * \
        input_dim[2] if isinstance(input_dim, tuple) else input_dim
    reader_test = init_reader(
        test_file, size, model.number_labels, is_training=False)
    eval_input_map = {
        model.input: reader_test.streams.features, model.label: reader_test.streams.labels}
    data = reader_test.next_minibatch(
        batch_size, input_map=eval_input_map)
    img_label = data[model.label].asarray()
    img_data = data[model.input].asarray()
    if isinstance(input_dim, tuple):
        img_data = np.reshape(img_data, (batch_size, 1, 28, 28))
    predicted_label_prob = [out.eval(img_data[i])
                            for i in range(len(img_data))]
    pred = [np.argmax(predicted_label_prob[i])
            for i in range(len(predicted_label_prob))]
    gtlabel = [np.argmax(img_label[i]) for i in range(len(img_label))]
    print("Label    :", gtlabel)
    print("Predicted:", pred)
