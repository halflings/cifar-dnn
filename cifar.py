import os
import pickle

import numpy as np

DATA_DIR = 'data'
CIFAR_DIR = os.path.join(DATA_DIR, 'cifar-10-batches-py')
NUM_BATCHES = 5
IMAGE_SIZE = 32
NUM_CHANNELS = 3


def unpickle_file(filename):
    with open(os.path.join(CIFAR_DIR, filename), mode='rb') as file:
        return pickle.load(file, encoding='bytes')


def convert_images(data):
    data = data.reshape([-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE])
    data = data.transpose([0, 2, 3, 1])
    data = data / 255.0
    return data


def load_dataset():
    metadata = unpickle_file('batches.meta')
    label_names, batch_size = metadata[b'label_names'], metadata[b'num_cases_per_batch']
    num_images = NUM_BATCHES * batch_size
    images = np.zeros(shape=[num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], dtype=np.float)
    labels = np.zeros(shape=[num_images], dtype=np.int)
    for batch_index in range(NUM_BATCHES):
        batch = unpickle_file('data_batch_{}'.format(batch_index + 1))
        begin_i, end_i = batch_index * batch_size, (batch_index + 1) * batch_size
        images[begin_i:end_i] = convert_images(batch[b'data'])
        labels[begin_i:end_i] = batch[b'labels']
    return images, labels, label_names


if __name__ == '__main__':
    load_dataset()
