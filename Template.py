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
            List containing all the train images
        list
            List containing all train images label
        list
            List containing all train images indexes
    '''
    faces_list = []
    labels_list = []
    indexes_list = []
    for index, label in enumerate(os.listdir(path)):
        labels_list.append(label)
        for image in os.listdir(path + '/' + label):
            faces_list.append(cv2.imread(path + '/' + label + '/' + image))
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
            List containing all the test subdirectories
    '''
    return os.listdir(path)


def get_all_test_images(path):
    '''
        To load a list of test images from given path list. Resize image height 
        to 200 pixels and image width to the corresponding ratio for train images

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
        full_image_path = path + '/' + image
        image_bgr = cv2.imread(full_image_path)
        image_shape = image_bgr.shape
        image_resized = cv2.resize(
            image_bgr, (int(200 * image_shape[1] / image_shape[0]), 200), interpolation=cv2.INTER_AREA)
        images_list.append(image_resized)

    return images_list


def detect_faces_and_filter(faces_list, labels_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is not equals to one

        Parameters
        ----------
        faces_list : list
            List containing all loaded images
        labels_list : list
            List containing all image classes labels

        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            list containing image gray face location
        list
            List containing all filtered image classes label
    '''
    face_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')

    grayed_images_list = []
    grayed_labels_list = []
    grayed_images_location = []

    for index, image in enumerate(faces_list):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(
            img_gray, scaleFactor=1.2, minNeighbors=8)

        if len(detected_faces) < 1:
            continue

        for face_rect in detected_faces:
            x, y, w, h = face_rect
            face_img = img_gray[y:y+w, x:x+h]
            grayed_images_list.append(face_img)
            grayed_images_location.append(face_rect)
            if labels_list is not None:
                grayed_labels_list.append(labels_list[index])

    return grayed_images_list, grayed_images_location, grayed_labels_list


def train(grayed_images_list, grayed_labels_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        grayed_images_list : list
            List containing all filtered and cropped face images in grayscale
        grayed_labels : list
            List containing all filtered image classes label

        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(grayed_images_list, np.array(grayed_labels_list))
    return recognizer


def predict(recognizer, gray_test_image_list):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        gray_test_image_list : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    predicted_test_image_list = []
    for image in gray_test_image_list:
        result, _ = recognizer.predict(image)
        predicted_test_image_list.append(result)

    return predicted_test_image_list


def check_attandee(predicted_name, room_number):
    '''
        To check the predicted user is in the designed room or not

        Parameters
        ----------
        predicted_name : str
            The name result from predicted user
        room_number : int
            The room number that the predicted user entered

        Returns
        -------
        bool
            If predicted user entered the correct room return True otherwise False
    '''
    if room_number == 1:
        if predicted_name in ['Elon Musk', 'Steve Jobs', 'Benedict Cumberbatch', 'Donald Trump']:
            return True
        else:
            return False
    if room_number == 2:
        if predicted_name in ['IU', 'Kim Se Jeong', 'Kim Seon Ho', 'Rich Brian']:
            return True
        else:
            return False


def write_prediction(predict_results, test_image_list, test_faces_rects, train_names, room):
    '''
        To draw prediction and validation results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories
        room: int
            The room number

        Returns
        -------
        list
            List containing all test images after being drawn with
            its prediction and validation results
    '''
    result_list = []
    for index, result in enumerate(predict_results):
        x, y, w, h = test_faces_rects[index]
        isValidRoom = check_attandee(train_names[result], room)
        if isValidRoom:
            cv2.rectangle(test_image_list[index],
                          (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(test_image_list[index], str(
                train_names[result] + ' - '+'Present'), (x-50, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        else:
            cv2.rectangle(test_image_list[index],
                          (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(test_image_list[index], str(
                train_names[result] + ' - '+"Shouldn't be here"), (x-50, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        result_list.append(test_image_list[index])

    return test_image_list


def combine_and_show_result(room, predicted_test_image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        room : str
            The room number in string format (e.g. 'Room 1')
        predicted_test_image_list : nparray
            Array containing image data
    '''
    result_image = np.concatenate(predicted_test_image_list, axis=1)
    cv2.imshow(str(room), result_image)
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
