import cntk as C

def mse(model, input):
    input_scale = input
    error = C.ops.reshape(C.ops.minus(model, input_scale),
                          (input.shape[1] * input.shape[2]))
    sq_err = C.ops.element_times(error, error)
    mse = C.ops.reduce_mean(sq_err)
    rmse_function = C.ops.sqrt(mse)
    return rmse_function


def cross_entropy_with_softmax(model, label):
    ce = C.cross_entropy_with_softmax(model, label)
    return ce


def classification_error(model, label):
    c_error = C.classification_error(model, label)
    return c_error
