import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Flatten, BatchNormalization, Convolution2D
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential, load_model
from keras.regularizers import l2
from driving_data import generator, load, summary_stats, pre_process

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('datadir', './data/', 'Root of the data directory')
flags.DEFINE_string('log_file', 'driving_log.csv', 'The metadata log csv file')
flags.DEFINE_string('model_file', 'model.h5', 'Saved model file')
flags.DEFINE_string('train_file', 'train_file.p', 'The pickle file for training data')
flags.DEFINE_string('valid_file', 'valid_file.p', 'The metadata for validation data')

flags.DEFINE_integer('batch_size', 64, 'The batch size for the generator')

flags.DEFINE_integer('top_crop', 60, 'Size of the image to crop from top')
flags.DEFINE_integer('bottom_crop', 20, 'Size of the image to crop from bottom')
flags.DEFINE_integer('sides_crop', 20, 'Size of the image to crop from bottom')

flags.DEFINE_float('zero_keep', 0.15, '% of the data to keep for zero steering angles')

flags.DEFINE_integer('epochs', 17, 'The number of epochs to train the fine tuning model')

zero_keep = FLAGS.zero_keep

batch_size = FLAGS.batch_size
epochs = FLAGS.epochs

data_dir = FLAGS.datadir
log_file = FLAGS.log_file
model_file = FLAGS.model_file
train_file = FLAGS.train_file
valid_file = FLAGS.valid_file

angle_correction = 0.1
top_crop = FLAGS.top_crop
bottom_crop = FLAGS.bottom_crop
sides_crop = FLAGS.sides_crop


def driving_model(input_shape):
    model = Sequential()

    model.add(Convolution2D(24, 8, 8, input_shape=input_shape, subsample=(2, 2), name='conv1'))
    model.add(BatchNormalization())
    model.add(Activation('elu', name='relu_1'))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), name='conv2'))
    model.add(BatchNormalization())
    model.add(Activation('elu', name='relu_2'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), name='conv3'))
    model.add(Activation('elu', name='relu_3'))

    model.add(Convolution2D(64, 3, 3, name='conv4'))
    model.add(Activation('elu', name='relu_4'))

    model.add(Convolution2D(64, 3, 3, name='conv5'))
    model.add(Activation('elu', name='relu_5'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(256, W_regularizer=l2(0.01), name='fc1'))
    model.add(Activation('elu', name='relu_7'))
    model.add(Dropout(0.4))

    model.add(Dense(100, W_regularizer=l2(0.01), name='fc3'))
    model.add(Activation('elu', name='relu_8'))
    model.add(Dropout(0.4))

    model.add(Dense(10, W_regularizer=l2(0.01), name='fc4'))
    model.add(Activation('elu', name='relu_9'))
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='linear', name='output'))

    return model


if __name__ == '__main__':

    raw_train_data, raw_valid_data = load(zero_keep, batch_size, data_dir,
                                          log_file, model_file, train_file,
                                          valid_file, top_crop, bottom_crop, sides_crop)

    image_shape = summary_stats(raw_train_data, raw_valid_data)
    print('Using epochs', epochs)

    model = driving_model((image_shape[0] - top_crop - bottom_crop, image_shape[1] - sides_crop * 2, image_shape[2]))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    if os.path.exists(model_file):
        model = load_model(model_file)

    history_object = model.fit_generator(generator(raw_train_data, batch_size=batch_size),
                                         samples_per_epoch=raw_train_data.shape[0],
                                         nb_epoch=epochs,
                                         validation_data=generator(raw_valid_data),
                                         nb_val_samples=raw_valid_data.shape[0])

    model.save(model_file)

    # Save the history object
    with open('history_object', 'wb') as pickle_file:
        pickle.dump(history_object.history, pickle_file)

    # predict steering angle for the first sample
    image = cv2.imread(data_dir + raw_train_data.loc[0][0])
    image = pre_process(image)
    image_shape = np.array(image).shape
    print('Image shape', image_shape)

    predicted_steering_angle = model.predict(image[None, :, :, :], batch_size=1)
    print('Predicted...', predicted_steering_angle)
    print('Actual...', raw_train_data.loc[0][1])

    with open('history_object', 'rb') as file:
        history_object = pickle.load(file)

        print(history_object.keys())

        print(history_object)
        # plot the training and validation loss for each epoch
        plt.plot(history_object['loss'])
        plt.plot(history_object['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()
    #
    # idx = 0
    # for images, angles in generator(raw_train_data[0:10], shuffle_data=False):
    #     print('images shape from generator', images.shape)
    #     print('angles shape from generator', angles.shape)
    #     idx += images.shape[0]
    #
    #     if idx >= raw_train_data[0:10].shape[0]:
    #         break
    # print('Total count processed', idx)