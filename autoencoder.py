# Import the relevant modules
from __future__ import print_function
#import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

# Import CNTK related modules
import cntk as C
from cntk.logging import ProgressPrinter
#from cntk.logging.graph import plot, find_by_name
from cntk.device import try_set_default_device, gpu, cpu
from cntk.layers import default_options, Dense
from cntk.io import StreamConfiguration, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.io import MinibatchSource, CTFDeserializer
from cntk.ops.functions import CloneMethod
from models.autoencoder import Autoencoder
from utils.losserrorfunctions.evaluationfunctions import mse


# Hard-coding the use of CPU
# TODO: Make this dynamic.
C.device.try_set_default_device(C.device.gpu(0))

def normalization(x):
    return x / 255

def create_reader(path, is_training, input_dim, num_label_classes):

    ctf = C.io.CTFDeserializer(path, C.io.StreamDefs(
          labels=C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
          features=C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)))

    return C.io.MinibatchSource(ctf,
        randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

def train(reader, model, loss_function, error_function, input_map, num_sweeps_to_train_with = 10, num_samples_per_sweep = 6000, minibatch_size = 64, learning_rate = 0.2):    
    # Instantiate the trainer object to drive the model training    
    lr_schedule = C.learning_parameter_schedule(learning_rate)
    learner = C.sgd(model.parameters, lr_schedule)

    # Print progress
    progress_printer_stdout = ProgressPrinter(freq=minibatch_size)

    # Instantiate trainer
    trainer = C.Trainer(model, (loss_function, error_function), [learner], progress_writers=progress_printer_stdout)

    # Start a timer
    start = time.time()
    aggregate_metric = 0
    total_samples = 0
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

    for i in range(0, int(num_minibatches_to_train)):
        # Read a mini batch from the training data file
        data = reader.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(data)
        samples = trainer.previous_minibatch_sample_count
        aggregate_metric += trainer.previous_minibatch_evaluation_average * samples
        total_samples += samples       

    # Print training time
    print("Training took {:.1f} sec".format(time.time() - start))
    print("Average error: {0:.2f}%".format((aggregate_metric * 100.0) / (total_samples)))

    return trainer

def test(reader, trainer, test_minibatch_size = 512, num_samples = 10000):
    # Test the model
    test_input_map = {
        labels  : reader.streams.labels,
        features  : reader.streams.features
    }

    # Test data for trained model  
    num_minibatches_to_test = num_samples // test_minibatch_size

    test_result = 0.0

    for i in range(num_minibatches_to_test):
        # We are loading test data in batches specified by test_minibatch_size
        # Each data point in the minibatch is a MNIST digit image of 784 dimensions
        # with one pixel per dimension that we will encode / decode with the
        # trained model.
        data = reader.next_minibatch(test_minibatch_size, input_map=test_input_map)
        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    print("Average test error: {0:.2f}%".format(test_result * 100.0 / num_minibatches_to_test))

def train_autoencoder():
    # Instantiate the model
    # Normalization function: We will scale the input image pixels within 0-1 range by dividing all input value by 255.
    autoencoder_definition = Autoencoder(input_dim = features, num_output_classes = num_output_classes, transformation = normalization)
    autoencoder_model = autoencoder_definition.create_autoencoder()
    
    reader_train = create_reader(train_file, True, input_dim, num_output_classes)

    # Train Autoencoder
    # Map the data streams to the input.
    # Instantiate the loss and error function.
    loss_function = mse(autoencoder_model, normalization(features))
    error_function = mse(autoencoder_model, normalization(features))

    input_map={
        features : reader_train.streams.features
    }
    
    train(reader=reader_train, model=autoencoder_model, loss_function=loss_function, error_function=error_function, input_map=input_map,
          num_sweeps_to_train_with = 100, num_samples_per_sweep = 2000, minibatch_size = 10, learning_rate = 0.02)

    autoencoder_model.save('autoencoder.model')

    return autoencoder_definition

def train_classifier(autoencoder_definition: Autoencoder):
    # Get the encoded layer, freeze its weights, add the classification and fine tune layers and train again
    encoded_model = autoencoder_definition.encoded_model
    feature_node = find_by_name(encoded_model, 'features')
    cloned_layers = C.combine([encoded_model]).clone(CloneMethod.freeze, {feature_node: features})

    classifier = autoencoder_definition.classifier
    full_model = classifier(cloned_layers)

    # Needs GraphViz. Ensure to add the path of Graphviz (anaconda folder/envs/'environment name'/library/bin/graphviz) into the system environment variables
    # plot_path = "full_model.png"
    # plot(full_model, plot_path)

    reader_train = create_reader(train_file, True, input_dim, num_output_classes)

    # Train Classifier
    # Instantiate the loss and error function.
    loss_function = C.cross_entropy_with_softmax(full_model, labels)
    error_function = C.classification_error(full_model, labels)

    input_map={
        labels : reader_train.streams.labels,
        features : reader_train.streams.features
    }

    trainer = train(reader=reader_train, model=full_model, loss_function=loss_function, error_function=error_function, input_map=input_map,
          num_sweeps_to_train_with = 100, num_samples_per_sweep = 2000, minibatch_size = 80, learning_rate = 0.02)

    full_model.save('full_model.model')

    return trainer

def test_model(trainer):
    reader_test = create_reader(test_file, False, input_dim, num_output_classes)

    test(reader=reader_test, trainer=trainer, test_minibatch_size = 10, num_samples = 500)

if __name__ == "__main__":
    features_shape = (1, 512, 512)
    input_dim = 512 * 512
    num_output_classes = 5
    features = C.input_variable(features_shape, name='features')
    labels = C.input_variable(num_output_classes, name='labels')

    train_folder = os.path.join("datasets", "nist_sd4", "files")
    train_file=os.path.join(train_folder, "train")
    test_folder = os.path.join("datasets", "nist_sd4", "files")
    test_file=os.path.join(test_folder, "test")

    #Train autoencoder
    autoencoder_definition = train_autoencoder()

    #Train classificator
    trainer = train_classifier(autoencoder_definition)

    #Test our model
    test_model(trainer)

