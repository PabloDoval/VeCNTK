import cntk

def swish(input_x):
    return input_x * cntk.ops.sigmoid(input_x)