import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

zero_keep = 0.15

batch_size = 64

data_dir = 'data/'
log_file = 'driving_log.csv'
model_file = 'model.h5'
train_file = 'train_file.p'
valid_file = 'valid_file.p'

angle_correction = 0.1
top_crop = 60
bottom_crop = 20
sides_crop = 20


def balanced_subsample(y):
    index_range = range(y.shape[0])
    size = int(np.ceil(y.shape[0] * zero_keep))
    indexes = np.random.choice(index_range, size=size, replace=False)
    return y[indexes]


def load(zero_keep, batch_size, datadir, log_file, model_file, train_file,
         valid_file, top_crop, bottom_crop, sides_crop):

    zero_keep = zero_keep

    batch_size = batch_size

    data_dir = datadir
    log_file = log_file
    model_file = model_file
    train_file = train_file
    valid_file = valid_file

    top_crop = top_crop
    bottom_crop = bottom_crop
    sides_crop = sides_crop

    # Load data from the csv file and split into training
    # and validation set
    data = []
    y = []
    if not os.path.isfile(train_file) and not os.path.isfile(valid_file):
        drive_info = pd.read_csv(data_dir + log_file, header=1, usecols=[0, 1, 2, 3, 4, 5, 6]).as_matrix();

        # Only keep sample of zero angle records
        filtered_data = drive_info[abs(drive_info[:, 3]) > 0.01]
        zero_data = drive_info[abs(drive_info[:, 3]) <= 0.01]
        zero_samples = balanced_subsample(zero_data)
        raw_data = np.row_stack((filtered_data, zero_samples))
        print('Raw training data after balancing', raw_data.shape)

        raw_train_data, raw_valid_data = train_test_split(raw_data, test_size=0.3, random_state=0)

        # augment the training data
        train_data = augment_data(raw_train_data)
        train_data.to_pickle(train_file)
        valid_data = reshape_data(raw_valid_data)
        valid_data.to_pickle(valid_file)
    else:
        train_data = pd.read_pickle(train_file)
        valid_data = pd.read_pickle(valid_file)

    print('Raw train data after augmentation', train_data.shape)
    print('Raw valid data', valid_data.shape)

    return train_data, valid_data


def summary_stats(train_entries, validation_entries):
    print('Training data size : ', train_entries.shape)
    print('Validation data size : ', validation_entries.shape)
    print('Top 5 rows of the data ', train_entries[0:5])
    print()
    image = cv2.imread(data_dir + train_entries.loc[0][0])
    image_shape = np.array(image).shape
    print('Image shape', image_shape)

    return image_shape


def augment_data(raw_data):
    augmented_data = pd.DataFrame(columns=('path', 'angle', 'flip', 'trans'))

    flipped_count = 0
    trans_count = 0

    idx = 0
    for row in raw_data:
        angle = row[3]

        # Center camera
        augmented_data.loc[idx] = create_row(row[0], angle)
        idx += 1

        if abs(angle) > 0.01:
            augmented_data.loc[idx] = create_row(row[0], angle, flipped=True)
            idx += 1
            flipped_count += 1

        augmented_data.loc[idx] = create_row(row[0], angle, trans=True)
        idx += 1
        trans_count += 1

        # Left camera
        augmented_data.loc[idx] = create_row(row[1], get_left_right_angle(row[3], True))
        idx += 1

        if abs(angle) > 0.01:
            augmented_data.loc[idx] = create_row(row[1], -get_left_right_angle(row[3], True), flipped=True)
            idx += 1
            flipped_count += 1

        augmented_data.loc[idx] = create_row(row[1], angle, trans=True)
        idx += 1
        trans_count += 1

        # right camera
        augmented_data.loc[idx] = create_row(row[2], get_left_right_angle(row[3], False))
        idx += 1

        if abs(angle) > 0.01:
            augmented_data.loc[idx] = create_row(row[2], -get_left_right_angle(row[3], False), flipped=True)
            idx += 1
            flipped_count += 1

        augmented_data.loc[idx] = create_row(row[2], angle, trans=True)
        idx += 1
        trans_count += 1

    print('Flipped count', flipped_count)
    print('Translatation count', trans_count)

    return augmented_data


# Translation function taken from web cedits - Vivek Yadav
def trans_image(image, steer, trans_range = 100):
    # Translation
    tr_x = trans_range*np.random.uniform() - trans_range/2
    steer_ang = steer + tr_x/trans_range * 2 * .2
    tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (320, 160))
    return image_tr, steer_ang


def reshape_data(raw_data):
    augmented_data = pd.DataFrame(columns=('path', 'angle', 'flip', 'trans'))
    idx = 0
    for row in raw_data:
        augmented_data.loc[idx] = create_row(row[0], row[3], False)
        idx += 1
    return augmented_data


def generator(data, shuffle_data=True, batch_size=16):
    num_data = len(data)

    while 1:  # Loop forever so the generator never terminates
        if shuffle_data:
            data = shuffle(data, random_state=12)
        for offset in range(0, num_data, batch_size):
            data_batch = data[offset:offset + batch_size]
            images = []
            angles = []

            for index, row in data_batch.iterrows():
                image = read_image(row[0])
                measurement = row[1]
                if row[2]:
                    image = cv2.flip(image, 1)
                if row[3]:
                    image, measurement = trans_image(image, measurement)

                image = pre_process(image)
                angles.append(measurement)
                images.append(image)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)

            yield X_train, y_train


def pre_process(image):
    image_shape = image.shape

    # Crop the image
    image = image[top_crop:image_shape[0] - bottom_crop, sides_crop: image_shape[1] - sides_crop]

    # normalize
    image = image / 127.5 - 1

    return image


def read_image(image_path):
    name = data_dir + str.strip(image_path)
    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
    return image


def get_left_right_angle(angle, isLeft):
    return (angle + angle_correction) if isLeft else (angle - angle_correction)


def create_row(path, angle, flipped=False, trans=False):
    newrow = [str.strip(path), angle, flipped, trans]
    return newrow


def augment_flip(image, angle):
    return np.fliplr(image), -angle