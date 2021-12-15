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
            faces_list.append(cv2.imread(path + '/' + label + '/' + image, 0))
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


def detect_faces_and_filter(faces_list, labels_list=None):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    grayed_images_list = []
    grayed_labels_list = []
    grayed_images_path_list = []

    for index, image in enumerate(faces_list):
        faces = face_cascade.detectMultiScale(image, 1.2, 5)

        if len(faces) == 1:
            for face in faces:
                x, y, w, h = face
                grayed_images_list.append(image[y: y+h, x: x+w])
                grayed_labels_list.append(labels_list[index])
                grayed_images_path_list.append(faces_list[index])

    return grayed_images_list, grayed_images_path_list, grayed_labels_list


def train(grayed_images_list, grayed_labels_list):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(grayed_images_list, np.array(grayed_labels_list))
    return face_recognizer


def predict(recognizer, gray_test_image_list):
    predict_results = []
    for image in gray_test_image_list:
        if image is not None:
            x, y, w, h = image  # ?
            predict_result, confidence = recognizer.predict(image)
            predict_results.append(predict_result)
            text = image[predict_result] + ': ' + str(confidence)
            cv2.putText(image, text, (x, y-10),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            cv2.imshow('Result', image)
    return predict_results


def check_attendee(predicted_name, room_number):
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
    predicted_test_image_list = []
    for i in range(len(predict_results)):
        ''''''
    return predicted_test_image_list


def combine_and_show_result(room, predicted_test_image_list):
    plt.figure(room)
    for idx, i in enumerate(predicted_test_image_list):
        plt.subplot(1, len(predicted_test_image_list), idx+1)
        plt.imshow(predicted_test_image_list[idx])
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


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
        predicted_test_image_list = write_prediction(
            predict_results, test_images_list, grayed_test_location, labels_list, index+1)
        combine_and_show_result(room, predicted_test_image_list)


if __name__ == "__main__":
    main()
