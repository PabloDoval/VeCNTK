from cntk import io


def init_reader(path, input_dim, num_label_classes, is_training=True):

    labelStream = io.StreamDef(
        field='labels', shape=num_label_classes, is_sparse=False)
    featureStream = io.StreamDef(
        field='features', shape=input_dim, is_sparse=False)
    deserializer = io.CTFDeserializer(path, io.StreamDefs(
        labels=labelStream, features=featureStream))
    return io.MinibatchSource(deserializer,
                              randomize=is_training,
                              max_sweeps=io.INFINITELY_REPEAT if is_training else 1)
