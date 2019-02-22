import keras
import pickle
from keras import backend as K
from enum import Enum
import types
import tempfile
import keras.models

class TestDimensionality(Enum):
    One = 1
    Multi = 2
    Bin = 3


class OnedimensionalTest(Enum):
    KS = 1


class MultidimensionalTest(Enum):
    MMD = 1
    KNN = 2
    FR = 3
    Energy = 4


class DimensionalityReduction(Enum):
    NoRed = 0
    PCA = 1
    SRP = 2
    UAE = 3
    TAE = 4
    BBSDs = 5
    BBSDh = 6
    Classif = 7


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


