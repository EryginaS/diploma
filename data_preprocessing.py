import os
import cv2
import numpy as np
import csv


def csv_reader(file_obj):
    """
    Read a csv file
    """
    label = []
    reader = csv.reader(file_obj)
    for row in reader:
        label.append(row)

        # print(" ".join(row))
    return np.array(label)


dir = "data/dataset"
files = os.listdir(dir)
print(files)
train_data = []
# [] - выборка   (number_of_exemple,height, width, channel) X
for file in os.listdir(dir):
    if file == 'BlurBody':
        for img in sorted(os.listdir(dir+'/BlurBody/img')):
            img_numpy = cv2.imread(os.path.join(dir+'/BlurBody/img/', img))
            train_data.append(img_numpy)



print(len(train_data))
X = np.array(train_data).astype('float32')/255.0
x_train = X[:250]
x_test = X[250:]

np.savez('x_train.npz', sequence_array=x_train)
np.savez('x_test.npz', sequence_array=x_test)
X_traint = np.load('x_train.npz')['sequence_array']
X_test = np.load('x_test.npz')['sequence_array']
print(X_test.shape, X_traint.shape)

# выборка Y

csv_path = "/home/sveta/diplloma/lstm_cnn/data/dataset/BlurBody/groundtruth_rect_BB.csv"
with open(csv_path, "r") as f_obj:
    label = csv_reader(f_obj)

y_train = label[:250]
y_test = label[250:]
np.savez('y_train.npz', sequence_array=y_train)
np.savez('y_test.npz', sequence_array=y_test)
Y_traint = np.load('y_train.npz')['sequence_array']
Y_test = np.load('y_test.npz')['sequence_array']
print(Y_test.shape, Y_traint.shape)
