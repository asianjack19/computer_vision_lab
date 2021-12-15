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
predicted_test_image_list.append(cv2.imread('Dataset/Train/Elon Musk/4.jpg'))

# plt.figure('Room Elon')
# plt.subplot(1, 4, 1)
# plt.imshow(predicted_test_image_list[0])
# plt.axis('off')
# plt.subplot(1, 4, 2)
# plt.imshow(predicted_test_image_list[1])
# plt.axis('off')
# plt.subplot(1, 4, 3)
# plt.imshow(predicted_test_image_list[2])
# plt.axis('off')
# plt.subplot(1, 4, 4)
# plt.imshow(predicted_test_image_list[3])
# plt.axis('off')
# plt.show()

plt.figure('Room Elon')
for idx, i in enumerate(predicted_test_image_list):
    plt.subplot(1, len(predicted_test_image_list), idx+1)
    plt.imshow(predicted_test_image_list[idx])
    plt.axis('off')
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
plt.show()
