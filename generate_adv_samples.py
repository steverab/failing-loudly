# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np
from tensorflow import set_random_seed
seed = 1
np.random.seed(seed)
set_random_seed(seed)

import sys
import os
import keras_resnet
import matplotlib.pyplot as plt

from tqdm import tqdm

from foolbox.models import KerasModel
from foolbox.attacks import FGSM

from keras.models import load_model

from data_utils import import_dataset, normalize_datapoints

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

# Load dataset identifier.
dataset = sys.argv[1]

# Define dataset-specific parameters.
if dataset == 'mnist':
    samp_shape = (28, 28)
    cmap = 'gray'
    max_epsilon = 1.0
elif dataset == 'cifar10':
    samp_shape = (32, 32, 3)
    cmap = None
    max_epsilon = 0.1
else:
    print('Dataset not supported')
    exit(0)

# Import dataset.
(_, _), (_, _), (X_te, y_te), orig_dims, _ = import_dataset(dataset, shuffle=True)
X_te = normalize_datapoints(X_te, 255.)

#external_dataset_path = './datasets/'
#x_test = np.loadtxt(external_dataset_path + 'cifar10_X_adversarial.csv', delimiter=',')
#y_test = np.loadtxt(external_dataset_path + 'cifar10_y_adversarial.csv', delimiter=',')
#print(x_test.shape)
#print(y_test.shape)
#print('-----------------------------------')
#np.save("%s/cifar10_X_adversarial.npy" % (external_dataset_path), x_test)
#np.save("%s/cifar10_y_adversarial.npy" % (external_dataset_path), y_test)
#x_test = np.load(external_dataset_path + 'cifar10_X_adversarial.npy')
#y_test = np.load(external_dataset_path + 'cifar10_y_adversarial.npy')
#print(x_test)
#print(y_test)
#exit(0)


# Load the Keras model.
model_path = './saved_models/' + dataset + '_standard_class_model.h5'
model = load_model(model_path, custom_objects=keras_resnet.custom_objects)

# Define plotting and plot samples.
plot_sample = True
plot_samples = 100

# Define plot path.
if plot_sample:
    path = './adv_samples/'
    path += dataset
    if not os.path.exists(path):
        os.makedirs(path)

# -------------------------------------------------
# ADVERSARIAL SAMPLE GENERATION
# -------------------------------------------------

# Create Foolbox model from Keras ResNet classifier and FGSM attack type.
foolbox_model = KerasModel(model, (0, 1))
attack = FGSM(foolbox_model)

# Turn all test set samples into adversarial samples.
for i in tqdm(range(len(X_te))):

    # Try to create an adversarial sample.
    adv_sample = attack(np.reshape(X_te[i], orig_dims), label=y_te[i], max_epsilon=max_epsilon)

    # In rare cases, sample generation might fail, which leaves adv_sample empty.
    if adv_sample is not None:

        # Successful adversarial samples are written back into the original matrix.
        X_te[i] = np.reshape(adv_sample, np.prod(orig_dims))

        # To get an impression of typical adversarial samples and to check whether their labels are indeed different,
        # we can plot the first plot_samples samples to disk.
        if plot_sample and i < plot_samples:

            # Forward-propagate the sample through the network to assess it's prediction.
            pred = model.predict(adv_sample.reshape(1, orig_dims[0], orig_dims[1], orig_dims[2]))
            pred = np.asscalar(np.argmax(pred, axis=1))

            # Plot sample to disk.
            fig = plt.imshow(adv_sample.reshape(samp_shape), cmap=cmap)
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.savefig("%s/%s_%s_%s.pdf" % (path, i, y_te[i], pred), bbox_inches='tight', pad_inches=0)

# Save results to datasets folder.
#np.savetxt("./datasets/%s_X_adversarial.csv" % dataset, X_te, delimiter=",")
#np.savetxt("./datasets/%s_y_adversarial.csv" % dataset, y_te, fmt='%5.0f', delimiter=",")
np.save("./datasets/%s_X_adversarial.npy" % dataset, X_te)
np.save("./datasets/%s_y_adversarial.npy" % dataset, y_te)