import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_train_image(path):
    faces_list = []
    labels_list = []
    indexes_list = []
    for index, label in enumerate(os.listdir(path)):
        for image in os.listdir(path + '/' + label):
            faces_list.append(cv2.imread(path + '/' + label + '/' + image))
            labels_list.append(label)
            indexes_list.append(index)

    return faces_list, labels_list, indexes_list


def get_all_test_folders(path):
    return os.listdir(path)


def get_all_test_images(path):
    images_list = []
    for image in os.listdir(path):
        image_shape = cv2.imread(path + '/' + image, 0).shape
        images_list.append(cv2.resize(cv2.imread(
            path + '/' + image, 0), (200, int(200 * image_shape[0] / image_shape[1]))))
    return images_list


def detect_faces_and_filter(faces_list, labels_list):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    grayed_images_list = []
    grayed_labels_list = []
    grayed_images_path_list = []
    for index, image in enumerate(faces_list):
        # image_shape = image.shape
        # image = cv2.resize(image, (200, int(200 * image_shape[0] / image_shape[1])))
        faces = face_cascade.detectMultiScale(image, 1.3, 5)
        if len(faces) == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for face in faces:
                x, y, w, h = face
                grayed_images_list.append(image[y:y + h, x:x + w])
                grayed_labels_list.append(labels_list[index])
                grayed_images_path_list.append(faces_list[index])

    # for index, image in enumerate(grayed_images_path_list):
    #     print(image)
    # for index, image in enumerate(grayed_images_list):
    #     cv2.imshow('image', image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # return grayed_images_list, faces, grayed_labels_list


def train(grayed_images_list, grayed_labels_list):
    '''
        To create and train face recognizer object

            Parameters
            ----------
            grayed_images_list : list
                List containing all filtered and cropped face images in grayscale
            grayed_labels_list : list
                List containing all image classes label

            Returns
            -------
            object
                Recognizer object after being trained with cropped face images
        '''
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(grayed_images_list, np.array(grayed_labels_list))
    return face_recognizer


def predict(recognizer, gray_test_image_list):
    predict_results = []
    for image in gray_test_image_list:
        if image is not None:
            predict_result = recognizer.predict(image)
            predict_results.append(predict_result)
    return predict_results


def check_attandee(predicted_name, room_number):
    '''
        To check the predicted user is in the designated room or not

        Parameters
        ----------
        predicted_name: str
            The name result from predicted user
        room_number: int
            The room number that the predicted user entered

        Returns
        -------
        bool
            If predicted user entered the correct room return True otherwise False
    '''


def write_prediction(predict_results, test_image_list, test_faces_rects, train_names, room):
    '''
        To draw prediction and validation results on the given test images

        Parameters
        ----------
        predict_results: list
            List containing all prediction results from given test faces
        test_image_list: list
            List containing all loaded test images
        test_faces_rects: list
            List containing all filtered faces location saved in rectangle
        train_names: list
            List containing the names of the train sub-directories
        room: int
            The room number

        Returns
        -------
        list
            List containing all test images after being drawn with
            its prediction and validation results
    '''


def combine_and_show_result(room, predicted_test_image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        room: str
            The room number in string format(e.g. 'Room 1')
        predicted_test_image_list: nparray
            Array containing image data
    '''
    horizontal = np.hstack(predicted_test_image_list)
    cv2.imshow(room, horizontal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # ERROR:
    # only integer scalar arrays can be converted to a scalar index
    #   NOTE: type of images <class 'numpy.ndarray'>
    # SOLUTION: convert to array(?) || find another command
    #   array == ndarray?


'''


You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''


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
    grayed_trained_images_list, _, grayed_trained_labels_list = detect_faces_and_filter(
        faces_list, indexes_list)
    recognizer = train(grayed_trained_images_list, grayed_trained_labels_list)
    '''
        Please modify test_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_path = "Dataset/Test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    test_images_folder = get_all_test_folders(test_path)
    for index, room in enumerate(test_images_folder):
        test_images_list = get_all_test_images(test_path + '/' + room)
        grayed_test_image_list, grayed_test_location, _ = detect_faces_and_filter(
            test_images_list)
        predict_results = predict(recognizer, grayed_test_image_list)
        for index, predict_result in enumerate(predict_results):
            print(predict_result)
        predicted_test_image_list = write_prediction(
            predict_results, test_images_list, grayed_test_location, labels_list, index+1)
        combine_and_show_result(room, predicted_test_image_list)


if __name__ == "__main__":
    main()
