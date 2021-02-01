import numpy as np


class Dataset:

    """
    Dataset is a class composing of
        a dataset : a list of numpy array
    """

    def __init__(self, dataset, label):
        self.datasets = dataset
        self.labels = label

