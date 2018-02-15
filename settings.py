MNIST_DATA_FOLDER = 'datasets/mnistdata/files/'
MINST_DOWNLOAD_INFO = {
    'BASE_URL': 'http://yann.lecun.com/exdb/mnist/',
    'TRAIN_DATA': {
        'NAME_IMAGE': 'train-images-idx3-ubyte.gz',
        'NAME_LABELS': 'train-labels-idx1-ubyte.gz',
        'SAMPLES': 60000
    },
    'TEST_DATA': {
        'NAME_IMAGE': 't10k-images-idx3-ubyte.gz',
        'NAME_LABELS': 't10k-labels-idx1-ubyte.gz',
        'SAMPLES': 10000
    }
}
NIST_SD4_DATA_FOLDER = 'datasets/nist_sd4/files/'
NIST_SD4_DOWNLOAD_INFO = {
    'DATASET_NAME': 'NIST-SD4',
    'FILE_NAME': 'NISTSpecialDatabase4GrayScaleImagesofFIGS.zip',
    'BASE_URL': 'https://s3.amazonaws.com/nist-srd/SD4/NISTSpecialDatabase4GrayScaleImagesofFIGS.zip',
    'TRAIN_SAMPLES': 50,
    'TEST_SAMPLES': 5
}
TENSOR_LOG_DIR = 'log'
