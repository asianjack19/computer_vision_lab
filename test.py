import os
import cv2

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

path = "Dataset/Test/Room 2"
image_list = []
for image in os.listdir(path):
    image_list.append(cv2.imread(path + '/' + image, 0))

# imshow image list
for i in range(len(image_list)):
    cv2.imshow(str(i), cv2.resize(
        image_list[i], (0, 0), fx=0.3, fy=0.3))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
