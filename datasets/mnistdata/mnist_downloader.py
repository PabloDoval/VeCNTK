import gzip
import struct
import numpy as np
from urllib.request import urlretrieve
from utils.files_path import join_paths, is_file
from settings import MNIST_DATA_FOLDER
from .mnist_files import savetxt
import os


class MnistDownloader:

    CHECK_IMAGE_NUMBER = 0x3080000
    CHECK_LABELS_NUMBER = 0x1080000
    IMAGE_SIZE = 28

    def __init__(self, mnist_info):
        self.__url_base = mnist_info.get('BASE_URL')
        self.__train_data = mnist_info.get('TRAIN_DATA')
        self.__test_data = mnist_info.get('TEST_DATA')

    def __load_images(self, download_info):
        download_url = join_paths(
            self.__url_base, download_info.get('NAME_IMAGE'))
        print('Downloading ' + download_url)
        gzfname, h = urlretrieve(download_url, './delete.me')
        print('Done.')
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack('I', gz.read(4))
                if n[0] != MnistDownloader.CHECK_IMAGE_NUMBER:
                    raise Exception('Invalid file: unexpected magic number.')
                n = struct.unpack('>I', gz.read(4))[0]
                if n != download_info.get('SAMPLES'):
                    raise Exception(
                        'Invalid file: expected {0} entries.'.format(download_info.get('SAMPLES')))
                crow = struct.unpack('>I', gz.read(4))[0]
                ccol = struct.unpack('>I', gz.read(4))[0]
                if crow != MnistDownloader.IMAGE_SIZE or ccol != MnistDownloader.IMAGE_SIZE:
                    raise Exception(
                        'Invalid file: expected 28 rows/cols per image.')
                res = np.fromstring(
                    gz.read(download_info.get('SAMPLES') * crow * ccol), dtype=np.uint8)
        finally:
            os.remove(gzfname)
        return res.reshape((download_info.get('SAMPLES'), crow * ccol))

    def __load_labels(self, download_info):
        download_url = join_paths(
            self.__url_base, download_info.get('NAME_LABELS'))
        print('Downloading ' + download_url)
        gzfname, h = urlretrieve(download_url, './delete.me')
        print('Done.')
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack('I', gz.read(4))
                if n[0] != MnistDownloader.CHECK_LABELS_NUMBER:
                    raise Exception('Invalid file: unexpected magic number.')
                n = struct.unpack('>I', gz.read(4))
                if n[0] != download_info.get('SAMPLES'):
                    raise Exception(
                        'Invalid file: expected {0} rows.'.format(download_info.get('SAMPLES')))
                res = np.fromstring(
                    gz.read(download_info.get('SAMPLES')), dtype=np.uint8)
        finally:
            os.remove(gzfname)
        return res.reshape((download_info.get('SAMPLES'), 1))

    def __download_data(self, train=True):
        download_info = self.__train_data if train else self.__test_data
        images = self.__load_images(download_info)
        labels = self.__load_labels(download_info)
        train_data = np.hstack((images, labels))
        return train_data

    def download_and_save_data(self):

        self.train_file = join_paths(MNIST_DATA_FOLDER, 'train.txt')
        if not is_file(self.train_file):
            train = self.__download_data(train=True)
            print('Writing train text file...')
            savetxt(self.train_file, train)

        self.test_file = join_paths(MNIST_DATA_FOLDER, 'test.txt')
        if not is_file(self.test_file):
            test = self.__download_data(train=False)
            print('Writing test text file...')
            savetxt(self.test_file, test)
