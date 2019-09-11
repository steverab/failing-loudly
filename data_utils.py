# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np
import scipy.io
from math import ceil

from keras.datasets import mnist, cifar10, cifar100, boston_housing, fashion_mnist
from keras.preprocessing.image import ImageDataGenerator

# -------------------------------------------------
# DATA UTILS
# -------------------------------------------------


def __unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def normalize_datapoints(x, factor):
    x = x.astype('float32') / factor
    return x


def random_shuffle(x, y):
    x, y = __unison_shuffled_copies(x, y)
    return x, y


def random_shuffle_and_split(x_train, y_train, x_test, y_test, split_index):
    x = np.append(x_train, x_test, axis=0)
    y = np.append(y_train, y_test, axis=0)

    x, y = __unison_shuffled_copies(x, y)

    x_train = x[:split_index, :]
    x_test = x[split_index:, :]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return (x_train, y_train), (x_test, y_test)


def import_dataset(dataset, shuffle=False):
    x_train, y_train, x_test, y_test = None, None, None, None
    external_dataset_path = './datasets/'
    nb_classes = 10
    if dataset == 'boston':
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(len(x_train), 28, 28, 1)
        x_test = x_test.reshape(len(x_test), 28, 28, 1)
    elif dataset == 'mnist_adv':
        (x_train, y_train), (_, _) = mnist.load_data()
        x_test = np.loadtxt(external_dataset_path + 'mnist_X_adversarial.csv', delimiter=',')
        y_test = np.loadtxt(external_dataset_path + 'mnist_y_adversarial.csv', delimiter=',')
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == 'cifar10_1':
        (x_train, y_train), (_, _) = cifar10.load_data()
        x_test = np.load(external_dataset_path + 'cifar10_1_v6_X.npy')
        y_test = np.load(external_dataset_path + 'cifar10_1_v6_y.npy')
        y_test = y_test.reshape((len(y_test),1))
    elif dataset == 'cifar10_adv':
        (x_train, y_train), (_, _) = cifar10.load_data()
        x_test = np.load(external_dataset_path + 'cifar10_adv_img.npy')
        y_test = np.load(external_dataset_path + 'cifar10_adv_label.npy')
        y_test = np.argmax(y_test, axis=1)
    elif dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(len(x_train), 28, 28, 1)
        x_test = x_test.reshape(len(x_test), 28, 28, 1)
    elif dataset == 'svhn':
        train = scipy.io.loadmat(external_dataset_path + 'svhn_train.mat')
        x_train = train['X']
        x_train = np.moveaxis(x_train, -1, 0)
        y_train = train['y']
        y_train[y_train == 10] = 0
        test = scipy.io.loadmat(external_dataset_path + 'svhn_test.mat')
        x_test = test['X']
        x_test = np.moveaxis(x_test, -1, 0)
        y_test = test['y']
        y_test[y_test == 10] = 0
    elif dataset == 'stl10':
        train = scipy.io.loadmat(external_dataset_path + 'stl10_train.mat')
        x_train = train['X']
        x_train = x_train.reshape(len(x_train), 96, 96, 3)
        y_train = train['y']
        y_train[y_train == 10] = 0
        test = scipy.io.loadmat(external_dataset_path + 'stl10_test.mat')
        x_test = test['X']
        x_test = x_test.reshape(len(x_test), 96, 96, 3)
        y_test = test['y']
        y_test[y_test == 10] = 0
    elif dataset == 'mnist_usps':
        data = scipy.io.loadmat(external_dataset_path + 'MNIST_vs_USPS.mat')
        x_train = data['X_src'].T
        x_test = data['X_tar'].T
        y_train = data['Y_src']
        y_test = data['Y_tar']

        y_train[y_train == 10] = 0
        y_test[y_test == 10] = 0

        x_train = x_train.reshape((len(x_train), 16, 16, 1))
        x_test = x_test.reshape((len(x_test), 16, 16, 1))

        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
    elif dataset == 'coil100':
        x = np.load(external_dataset_path + 'coil100_X.npy')
        y = np.load(external_dataset_path + 'coil100_y.npy')
        
        cats = 100
        nb_classes = cats
        samples_per_cat = 72
        
        x = x[:cats * samples_per_cat,:]
        y = y[:cats * samples_per_cat]
        
        img_size = 32
        img_channels = 3
        train_samples_per_cat = 72 * 2 // 3
        train_samples = train_samples_per_cat * cats
        test_samples_per_cat = samples_per_cat - train_samples_per_cat
        test_samples = test_samples_per_cat * cats
        
        x_train = np.ones((train_samples , img_size, img_size, img_channels)) * (-1)
        y_train = np.ones(train_samples) * (-1)
        x_test = np.ones((test_samples , img_size, img_size, img_channels)) * (-1)
        y_test = np.ones(test_samples) * (-1)
        
        i = 0
        j = 0
        while i < len(x):
            x_train[i:i+train_samples_per_cat] = x[j:j+train_samples_per_cat]
            y_train[i:i+train_samples_per_cat] = y[j:j+train_samples_per_cat]
            i = i + train_samples_per_cat
            j = j + samples_per_cat
            
        i = 0
        j = 0
        while i < len(x):
            x_test[i:i+test_samples_per_cat] = x[j+train_samples_per_cat:j+samples_per_cat]
            y_test[i:i+test_samples_per_cat] = y[j+train_samples_per_cat:j+samples_per_cat]
            i = i + test_samples_per_cat
            j = j + samples_per_cat

    if shuffle:
        (x_train, y_train), (x_test, y_test) = random_shuffle_and_split(x_train, y_train, x_test, y_test, len(x_train))

    if dataset == 'mnist_usps':
        x_test = x_test[:1000,:]
        y_test = y_test[:1000]    

    # Add 3-way split
    x_train_spl = np.split(x_train, [len(x_train) - len(x_test)])
    y_train_spl = np.split(y_train, [len(y_train) - len(y_test)])
    x_train = x_train_spl[0]
    x_val = x_train_spl[1]
    y_train = y_train_spl[0]
    y_val = y_train_spl[1]

    orig_dims = x_train.shape[1:]

    # Reshape to matrix form
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    y_train = y_train.reshape(len(y_train))
    y_val = y_val.reshape(len(y_val))
    y_test = y_test.reshape(len(y_test))
    
    print(orig_dims)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), orig_dims, nb_classes


# Merge clean and perturbed data based on given percentage.
def data_subset(x_clean, y_clean, x_altered, y_altered, delta=1.0):
    indices = np.random.choice(x_clean.shape[0], ceil(x_clean.shape[0] * delta), replace=False)
    indices_altered = np.random.choice(x_clean.shape[0], ceil(x_clean.shape[0] * delta), replace=False)
    x_clean[indices, :] = x_altered[indices_altered, :]
    y_clean[indices] = y_altered[indices_altered]
    return x_clean, y_clean, indices


# Perform image perturbations.
def image_generator(x, orig_dims, rot_range, width_range, height_range, shear_range, zoom_range, horizontal_flip, vertical_flip, delta=1.0):
    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta), replace=False)
    datagen = ImageDataGenerator(rotation_range=rot_range,
                                 width_shift_range=width_range,
                                 height_shift_range=height_range,
                                 shear_range=shear_range,
                                 zoom_range=zoom_range,
                                 horizontal_flip=horizontal_flip,
                                 vertical_flip=vertical_flip,
                                 fill_mode="nearest")
    x_mod = x[indices, :]
    for idx in range(len(x_mod)):
        img_sample = x_mod[idx, :].reshape(orig_dims)
        mod_img_sample = datagen.flow(np.array([img_sample]), batch_size=1)[0]
        x_mod[idx, :] = mod_img_sample.reshape(np.prod(mod_img_sample.shape))
    x[indices, :] = x_mod
    
    return x, indices


def gaussian_noise(x, noise_amt, normalization=1.0, clip=True):
    noise = np.random.normal(0, noise_amt / normalization, (x.shape[0], x.shape[1]))
    if clip:
        return np.clip(x + noise, 0., 1.)
    else:
        return x + noise


def gaussian_noise_subset(x, noise_amt, normalization=1.0, delta_total=1.0, clip=True):
    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta_total), replace=False)
    x_mod = x[indices, :]
    noise = np.random.normal(0, noise_amt / normalization, (x_mod.shape[0], x_mod.shape[1]))
    if clip:
        x_mod = np.clip(x_mod + noise, 0., 1.)
    else:
        x_mod = x_mod + noise
    x[indices, :] = x_mod
    return x, indices


# Remove instances of a single class.
def knockout_shift(x, y, cl, delta):
    del_indices = np.where(y == cl)[0]
    until_index = ceil(delta * len(del_indices))
    if until_index % 2 != 0:
        until_index = until_index + 1
    del_indices = del_indices[:until_index]
    x = np.delete(x, del_indices, axis=0)
    y = np.delete(y, del_indices, axis=0)
    return x, y


# Remove all classes except for one via multiple knock-out.
def only_one_shift(X, y, c):
    I = len(np.unique(y))
    i = 0
    while i < I:
        if i == c:
            i = i + 1
            continue
        X, y = knockout_shift(X, y, i, 1.0)
        i = i + 1
    return X, y


def adversarial_samples(dataset):
    x_test, y_test = None, None
    external_dataset_path = './datasets/'
    if dataset == 'mnist':
        x_test = np.load(external_dataset_path + 'mnist_X_adversarial.npy')
        y_test = np.load(external_dataset_path + 'mnist_y_adversarial.npy')
    elif dataset == 'cifar10':
        x_test = np.load(external_dataset_path + 'cifar10_X_adversarial.npy')
        y_test = np.load(external_dataset_path + 'cifar10_y_adversarial.npy')
    return x_test, y_test
