import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

# path = "Dataset/Test/Room 1"

# faces_list = []
# labels_list = []
# indexes_list = []
# for index, label in enumerate(os.listdir(path)):
#     for image in os.listdir(path + '/' + label):
#         faces_list.append(cv2.imread(path + '/' + label + '/' + image, 0))
#         labels_list.append(label)
#         indexes_list.append(index)

# # cast int to string in python

# for i in range(len(faces_list)):
#     cv2.imshow(labels_list[i]+str(indexes_list[i]), cv2.resize(
#         faces_list[i], (0, 0), fx=0.3, fy=0.3))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# images_list = []
# for image in os.listdir(path):
#     images_list.append(cv2.imread(path + '/' + image, 0))

# print all images list
# for i in range(len(images_list)):
#     cv2.imshow(str(i), cv2.resize(
#         images_list[i], (0, 0), fx=0.3, fy=0.3))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# path = "Dataset/Test/Room 2"
# image_list = []
# for image in os.listdir(path):
#     image_list.append(cv2.imread(path + '/' + image, 0))


predicted_test_image_list = []
predicted_test_image_list.append(cv2.imread('Dataset/Train/Elon Musk/1.jpg'))
predicted_test_image_list.append(cv2.imread('Dataset/Train/Elon Musk/2.jpg'))
predicted_test_image_list.append(cv2.imread('Dataset/Train/Elon Musk/3.jpg'))

# imshow image list
# for i in range(len(predicted_test_image_list)):
#     predicted_test_image_list[i] = cv2.resize(
#         predicted_test_image_list[i], (0, 0), None, 0.5, 0.5)
#     cv2.imshow(str(i), predicted_test_image_list[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

scale_percent = 50/100
# for i in range(len(predicted_test_image_list)):
#     width = int(predicted_test_image_list[i].shape[1] * scale_percent)
#     height = int(predicted_test_image_list[i].shape[0] * scale_percent)
#     dim = (width, height)
#     predicted_test_image_list[i] = cv2.resize(
#         predicted_test_image_list[i], dim, cv2.INTER_AREA)
#     cv2.imshow(str(i), predicted_test_image_list[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

for i in range(len(predicted_test_image_list)):
    predicted_test_image_list[i] = np.array(predicted_test_image_list[i])
    print(type(predicted_test_image_list[i]))
    # predicted_test_image_list[i] = predicted_test_image_list[i].resize(
    #     (900, 900))
    # cv2.imshow(str(i), predicted_test_image_list[i])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# combine_and_show_result("Room 1", predicted_test_image_list)
'''
    To show the final image that already combine into one image

    Parameters
    ----------
    room: str
        The room number in string format(e.g. 'Room 1')
    predicted_test_image_list: nparray
        Array containing image data
'''
horizontal = np.hstack(
    (predicted_test_image_list[0], predicted_test_image_list[1]))
cv2.imshow("Room 1", horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()

arr = np.array([1, 2, 3, 4, 5, 6])
hor = np.hstack(arr)
print(hor)
