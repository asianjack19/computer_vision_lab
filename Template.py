import os
import cv2
import numpy as np


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
        detected_face = face_cascade.detectMultiScale(image, 1.2, 5)

        if len(detected_face) == 1:
            for face_rect in detected_face:
                x, y, w, h = face_rect
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
            predict_result, _ = recognizer.predict(image)
            predict_results.append(predict_result)
    return predict_results


def check_attendee(predicted_name, room_number):
    if room_number == 1:
        if predicted_name != 'Elon Musk' and predicted_name != 'Steve Jobs' and predicted_name != 'Benedict Cumberbatch' and predicted_name != 'Donald Trump':
            return False
        else:
            return True
    elif room_number == 2:
        if predicted_name != 'IU' and predicted_name != 'Kim Se Jeong' and predicted_name != 'Kim Seon Ho' and predicted_name != 'Rich Brian':
            return False
        else:
            return True


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
    return predicted_test_image_list


def combine_and_show_result(room, predicted_test_image_list):
    h_min = min(img.shape[0] for img in predicted_test_image_list)
    h_min = int(h_min * 80/100)
    resized_image_list = [cv2.resize(img,
                                     (int(img.shape[1] * h_min / img.shape[0]),
                                      h_min), interpolation=cv2.INTER_CUBIC)
                          for img in predicted_test_image_list]
    cv2.imshow(room, resized_image_list)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
