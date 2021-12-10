

import os
import cv2
import numpy as np


def get_train_image(path):
    '''
        To get a list of train images, images label, and images index using the given path

            Parameters
            ----------
            path : str
                Location of train root directory

            Returns
            -------
            list
                List containing all train images
            list
                List containing all train images label
            list
                List containing all train images index
        '''
    faces_list = []
    labels_list = []
    indexes_list = []
    for index, label in enumerate(os.listdir(path)):
        for image in os.listdir(path + '/' + label):
            faces_list.append(cv2.imread(path + '/' + label + '/' + image, 0))
            labels_list.append(label)
            indexes_list.append(index)
    return faces_list, labels_list, indexes_list


def get_all_test_folders(path):
    '''
        To get a list of test subdirectories using the given path

            Parameters
            ----------
            path : str
                Location of test root directory

            Returns
            -------
            list
                List containing all test subdirectories
        '''
    return os.listdir(path)


def get_all_test_images(path):
    '''
        To load a list of test images from given path list. Resize image height to 200 pixels and image width to the corresponding ratio for train images

        Parameters
        ----------
        path : str
            Location of images root directory

        Returns
        -------
        list
            List containing all image that has been resized for each Test Folders
    '''
    images_list = []
    for image in os.listdir(path):
        images_list.append(cv2.imread(path + '/' + image, 0))
    return images_list


def main():
    '''
        Please modify train_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_path = "Dataset/Train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    faces_list, labels_list, indexes_list = get_train_image(train_path)


if __name__ == "__main__":
    main()
