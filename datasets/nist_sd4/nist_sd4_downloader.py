import zipfile
import os
import glob
import shutil
import numpy as np
import struct
import sys
from scipy.misc import imread
from datasets.nist_sd4.fingerprint import fingerprint
from settings import NIST_SD4_DATA_FOLDER

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

class NistSD4Downloader:
    """Downloads the dataset if needed, and copies the files to the appropiate directories."""

    def __init__(self, nist_sd4_info):
        self.url_base = nist_sd4_info.get('BASE_URL')
        self.dataset_name = nist_sd4_info.get('DATASET_NAME')
        self.destinationFolder = NIST_SD4_DATA_FOLDER
        self.fileName = os.path.join(self.destinationFolder, nist_sd4_info.get('FILE_NAME'))
        self.testFolder = os.path.join(self.destinationFolder, "testImgs")
        self.trainFolder = os.path.join(self.destinationFolder, "trainImgs")
        self.testSamples = nist_sd4_info.get('TEST_SAMPLES')
        self.trainSamples = nist_sd4_info.get('TRAIN_SAMPLES')


    def __downloadDataset(self):
        
        if not os.path.exists(os.path.join(self.destinationFolder, "test")):
            if not os.path.exists(self.fileName):
                os.makedirs(NIST_SD4_DATA_FOLDER)
                print("Downloading data from " + self.url_base + "...")
                urlretrieve(self.url_base, self.fileName)

            try:
                print("Extracting " + self.fileName + "...")
                with zipfile.ZipFile(self.fileName) as myzip:
                    myzip.extractall(self.destinationFolder)
                print("Distributing the Dataset...")
                self.__distributeDataset(self.destinationFolder, self.testFolder, self.trainFolder)
                print("Renaming the files...")
                self.__renameFiles(self.testFolder)
                self.__renameFiles(self.trainFolder)
            finally:
                os.remove(self.fileName)
            print("Done.")
        else:
            print("Data already available at " + self.destinationFolder + "/" + self.dataset_name)

    def __distributeDataset(self, destinationFolder, testFolder, trainFolder):
        """Creates the Test and Train directories if needed, and moves the images into them."""
        
        # Set up directories for test and training data sets
        if not os.path.exists(testFolder):
            os.makedirs(testFolder)
        if not os.path.exists(trainFolder):
            os.makedirs(trainFolder)

        # Generate list of directories
        dirs = []
        for i in range(0,8):
            dirs.append(os.path.join(destinationFolder, "NISTSpecialDatabase4GrayScaleImagesofFIGS\\sd04\\png_txt\\figs_" + str(i)))

        # Extract Test data
        files = os.listdir(dirs[0])

        for filename in files:
            shutil.copy(os.path.join(dirs[0], filename), testFolder)
        shutil.rmtree(dirs[0])

        # Extract Train data
        for i in range(1,8):

            files = os.listdir(dirs[i])
            for filename in files:
                shutil.copy(os.path.join(dirs[i], filename), trainFolder)
            shutil.rmtree(dirs[i])
        shutil.rmtree(os.path.join(destinationFolder, "NISTSpecialDatabase4GrayScaleImagesofFIGS"))

    def __renameFiles(self, folder):
        """Renames all files so the names hold the metadata information."""

        # Retrieve list of all text files and remove the txt files
        for filename in glob.glob(os.path.join(folder, "*.txt")):
            with open(filename, 'r') as file:
                metadata=file.read().replace('\n', '')
            ident = metadata[27:31]
            order = metadata[26].upper()
            finger = metadata[32:34]
            gender = metadata[8].upper()
            fingerprintClass = metadata[16].upper()
            fp = fingerprint(ident, order, finger, gender, fingerprintClass)

            # Remove the .txt file and rename the png
            os.remove(filename)
            pngName = filename.replace(".txt", ".png")
            newName = fp.id + "_" + fp.order + "_" + fp.finger + "_" + fp.gender + "_" + fp.fingerprintClass + ".png"
            newName = os.path.join(folder, newName)
            os.rename(pngName, newName)

    def __encodeFingerprintClassFromFileName(self, fileName):
        """Converts the fingerprintType into the one-hot encoding format."""
        fileName = os.path.basename(fileName)
        fingerprintClass = fileName[12]
    
        if fingerprintClass.upper() == "A":
            return np.array([1, 0, 0, 0, 0])
        elif fingerprintClass.upper() == "L":
            return np.array([0, 1, 0, 0, 0])
        elif fingerprintClass.upper() == "R":
            return np.array([0, 0, 1, 0, 0])
        elif fingerprintClass.upper() == "T":
            return np.array([0, 0, 0, 1, 0])
        elif fingerprintClass.upper() == "W":
            return np.array([0, 0, 0, 0, 1])

    def __imagesToVector(self, sourceFolder, imagesToProcess = 0):
        """Converts the PNG images to a one-dimensional vector, with an initial one-hot encodding for the feature"""

        # Iterate all the images allowed by the imagesToProcess parameter
        files = glob.glob(os.path.join(sourceFolder, "*.png"))
        imgCont = 1
        if imagesToProcess == 0:
            numRows = len(files)
        else:
            numRows = min(len(files),imagesToProcess)
        numFeatures = 5
        res = np.ndarray(shape = (numRows, 512*512 + numFeatures), dtype = np.uint8)

        for file in files:

            if imgCont > imagesToProcess:
                break

            # Extract the features
            features = self.__encodeFingerprintClassFromFileName(file)

            # Extract the bytes of the 512 x 512 images as a vector
            img = imread(file, flatten = True)
            flattened = img.ravel()
            res[imgCont-1] = np.concatenate([features,flattened])
            
            imgCont += 1

        return res

    def __saveToTxt(self, filename, vector):
        """Saves the image vectors to One-Hot encoding text files, compatible with the CNTK text reader"""

        print("Saving file: ", filename)
        with open(filename, "w") as f:
            # labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            for row in vector:
                label_str = ' '.join(row.astype(str)[0:5])
                feature_str = ' '.join(row.astype(str)[5:])
                f.write("|labels {} |features {}\n".format(label_str, feature_str))

    def __processImages(self, sourceFolder, outputFileName, imagesToProcess = 0):
        images = self.__imagesToVector(sourceFolder, imagesToProcess)
        self.__saveToTxt(outputFileName, images)

    def download_and_save_data(self):
        self.__downloadDataset()

    def process(self):
        self.__processImages(self.trainFolder, os.path.join(NIST_SD4_DATA_FOLDER, 'train'), self.trainSamples)
        self.__processImages(self.testFolder, os.path.join(NIST_SD4_DATA_FOLDER, 'test'), self.testSamples)