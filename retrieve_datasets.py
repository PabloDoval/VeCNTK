from datasets.mnistdata.mnist_downloader import MnistDownloader
from datasets.mnistdata.ctf_reader import init_reader
from datasets.nist_sd4.nist_sd4_downloader import NistSD4Downloader

from settings import MINST_DOWNLOAD_INFO, NIST_SD4_DOWNLOAD_INFO


if __name__ == '__main__':

    mnist_downloader = MnistDownloader(MINST_DOWNLOAD_INFO)
    mnist_downloader.download_and_save_data()

    nist_sd4_downloader = NistSD4Downloader(NIST_SD4_DOWNLOAD_INFO)
    nist_sd4_downloader.download_and_save_data()
    nist_sd4_downloader.process()