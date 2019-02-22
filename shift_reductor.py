import numpy as np

from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection

from keras.layers import Input, Dense, Dropout, Activation, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

from keras import optimizers

from enum import Enum
from shared_utils import *
import os

import keras_resnet.models


class ShiftReductor:

    def __init__(self, X, y, X_val, y_val, dr_tech, orig_dims, datset, dr_amount=None, var_ret=0.8):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.dr_tech = dr_tech
        self.orig_dims = orig_dims
        self.datset = datset
        self.mod_path = None

        if dr_amount is None:
            pca = PCA(n_components=var_ret, svd_solver='full')
            pca.fit(X)
            self.dr_amount = pca.n_components_
        else:
            self.dr_amount = dr_amount

    def fit_reductor(self):
        if self.dr_tech == DimensionalityReduction.PCA:
            return self.principal_components_anaylsis()
        elif self.dr_tech == DimensionalityReduction.SRP:
            return self.sparse_random_projection()
        elif self.dr_tech == DimensionalityReduction.UAE:
            self.mod_path = './saved_models/' + self.datset + '_untr_autoencoder_model.h5'
            if os.path.exists(self.mod_path):
                return load_model(self.mod_path)
            return self.autoencoder(train=False)
        elif self.dr_tech == DimensionalityReduction.TAE:
            self.mod_path = './saved_models/' + self.datset + '_autoencoder_model.h5'
            if os.path.exists(self.mod_path):
                return load_model(self.mod_path)
            return self.autoencoder(train=True)
        elif self.dr_tech == DimensionalityReduction.BBSDs:
            self.mod_path = './saved_models/' + self.datset + '_standard_class_model.h5'
            if os.path.exists(self.mod_path):
                return load_model(self.mod_path, custom_objects=keras_resnet.custom_objects)
            return self.neural_network_classifier(train=True)
        elif self.dr_tech == DimensionalityReduction.BBSDh:
            self.mod_path = './saved_models/' + self.datset + '_standard_class_model.h5'
            if os.path.exists(self.mod_path):
                return load_model(self.mod_path, custom_objects=keras_resnet.custom_objects)
            return self.neural_network_classifier(train=True)

    def reduce(self, model, X):
        if self.dr_tech == DimensionalityReduction.PCA or self.dr_tech == DimensionalityReduction.SRP:
            return model.transform(X)
        elif self.dr_tech == DimensionalityReduction.UAE or self.dr_tech == DimensionalityReduction.TAE or self.dr_tech == DimensionalityReduction.BBSDs:
            X = X.reshape(len(X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2])
            pred = model.predict(X)
            pred = pred.reshape((len(pred), np.prod(pred.shape[1:])))
            return pred
        elif self.dr_tech == DimensionalityReduction.NoRed:
            return X
        elif self.dr_tech == DimensionalityReduction.BBSDh:
            X = X.reshape(len(X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2])
            pred = model.predict(X)
            pred = np.argmax(pred, axis=1)
            return pred

    def sparse_random_projection(self):
        srp = SparseRandomProjection(n_components=self.dr_amount)
        srp.fit(self.X)
        return srp

    def principal_components_anaylsis(self):
        pca = PCA(n_components=self.dr_amount)
        pca.fit(self.X)
        return pca

    def autoencoder(self, train=False):
        X = self.X.reshape(len(self.X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2])

        input_img = Input(shape=self.orig_dims)

        if self.datset == 'mnist' or self.datset == 'fashion_mnist':
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
            encoded = MaxPooling2D((2, 2), padding='same')(x)

            x = Conv2D(2, (3, 3), activation='relu', padding='same')(encoded)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(16, (3, 3), activation='relu')(x)
            x = UpSampling2D((2, 2))(x)
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        elif self.datset == 'cifar10' or self.datset == 'svhn':

            x = Conv2D(64, (3, 3), padding='same')(input_img)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(16, (3, 3), padding='same')(x)
            x = Activation('relu')(x)
            encoded = MaxPooling2D((2, 2), padding='same')(x)

            x = Conv2D(16, (3, 3), padding='same')(encoded)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(64, (3, 3), padding='same')(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(3, (3, 3), padding='same')(x)
            decoded = Activation('sigmoid')(x)

        elif self.datset == 'mnist_usps':
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
            encoded = MaxPooling2D((2, 2), padding='same')(x)

            x = Conv2D(2, (3, 3), activation='relu', padding='same')(encoded)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


        encoder = Model(input_img, encoded)
        autoenc = Model(input_img, decoded)

        autoenc.compile(optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9), loss='binary_crossentropy')

        if train:
            autoenc.fit(self.X.reshape(len(self.X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2]), self.X.reshape(len(self.X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2]),
                        epochs=200,
                        batch_size=128,
                        validation_data=(self.X_val.reshape(len(self.X_val), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2]), self.X_val.reshape(len(self.X_val), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2])),
                        shuffle=True)
                        
        encoder.save(self.mod_path)

        return encoder

    def neural_network_classifier(self, train=True):
        D = self.X.shape[1]

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        early_stopper = EarlyStopping(min_delta=0.001, patience=10)
        batch_size = 128
        nb_classes = len(np.unique(self.y))
        epochs = 200
        y_loc = np_utils.to_categorical(self.y, 10)
        y_val_loc = np_utils.to_categorical(self.y_val, 10)

        model = keras_resnet.models.ResNet18(keras.layers.Input(self.orig_dims), classes=nb_classes)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9),
                      metrics=['accuracy'])

        model.fit(self.X.reshape(len(self.X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2]), y_loc,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(self.X_val.reshape(len(self.X_val), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2]), y_val_loc),
                  shuffle=True,
                  callbacks=[lr_reducer, early_stopper])
                  
        model.save(self.mod_path)

        return model
