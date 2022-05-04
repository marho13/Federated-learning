import os
import numpy as np  # linear algebra
import struct
from array import array
import torch
from torchvision.transforms import Resize, InterpolationMode
from os.path import join
import random
import matplotlib.pyplot as plt
import torch

def getFolders():
    output = []
    folder = "C:/Users/Martin/Downloads/Mnist/"

    files = os.listdir(folder)
    fileList = [(folder+f) for f in files]
    output.append(fileList[-2])
    output.append(fileList[-1])
    output.append(fileList[0])
    output.append(fileList[1])
    return output
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            # img = np.swapaxes(img, -1, 0)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

    def getDatasetRandomly(self, xTrain, yTrain, xTest, yTest):
        images_train = []
        titles_train = []
        images_test = []
        titles_test = []
        numTrain = [i for i in range(60000)]
        random.shuffle(numTrain)
        for r in numTrain: #For all of them (all 60 0000 and 10 000)
            images_train.append(xTrain[r])
            titles_train.append(yTrain[r])

        numTest = [j for j in range(10000)]
        random.shuffle(numTest)
        for r in numTest:
            images_test.append(xTest[r])
            titles_test.append(yTest[r])
        return images_train, titles_train, images_test, titles_test

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    # cols = 5
    # rows = int(len(images)/cols) + 1
    plt.figure(figsize=(224,224))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]

        # plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15)
        index += 1
        plt.show()
    # plt.plot()

def getMnist():
    fileList = getFolders()
    mnist_dataloader = MnistDataloader(fileList[0], fileList[1], fileList[2], fileList[3])
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    x_train, y_train, x_test, y_test = mnist_dataloader.getDatasetRandomly(x_train, y_train, x_test, y_test)
    x_train, y_train, x_test, y_test = torch.tensor(x_train).float(), torch.tensor(y_train), torch.tensor(x_test).float(), torch.tensor(y_test)
    # x_train = Resize((252, 252), InterpolationMode.BILINEAR)(x_train)
    # x_test = Resize((252, 252), InterpolationMode.BILINEAR)(x_test)
    return x_train, y_train, x_test, y_test



if __name__ == "__main__":
    xTrain, yTrain, xTest, yTest = getMnist()
    reshaper = Resize((224, 224), InterpolationMode.BILINEAR)
    xTrain = reshaper(xTrain)
    show_images(xTrain, yTrain)