from cntk import device
import cntk as cn


def set_devices():
    if (cn.__version__ != '2.3'):
        raise Exception('[ERROR]: Invalid CNTK Version')
    all_devices = device.all_devices()
    if all_devices[0].type() == device.DeviceKind.GPU:
        print('[INFO]: You computer''s GPU suport CUDA acceleration')
        device.try_set_default_device(device.gpu(0))
    else:
        print('[WARNING]: You computer''s GPU does not suport CUDA acceleration')
        device.try_set_default_device(device.cpu())
