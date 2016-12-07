import numpy as np
import data_classification_utils
from util import raiseNotDefined
import random
import util

class Perceptron(object):
    def __init__(self, categories, numFeatures):
        """categories: list of strings
           numFeatures: int"""
        self.categories = categories
        self.numFeatures = numFeatures

        """YOUR CODE HERE"""
        self.weights = {}
        for category in self.categories:
            self.weights[category] = np.zeros(self.numFeatures)


    def classify(self, sample):
        """sample: np.array of shape (1, numFeatures)
           returns: category with maximum score, must be from self.categories"""

        """YOUR CODE HERE"""
        vector = util.Counter()
        for category in self.categories:
            vector[category] = self.weights[category].dot(sample)
        return vector.argMax()


    def train(self, samples, labels):
        """samples: np.array of shape (numSamples, numFeatures)
           labels: list of numSamples strings, all of which must exist in self.categories
           performs the weight updating process for perceptrons by iterating over each sample once."""

        """YOUR CODE HERE"""
        for i, sample in enumerate(samples):
            true_label = labels[i]
            predicated_label = self.classify(sample)
            if true_label != predicated_label:
                self.weights[true_label] += sample
                self.weights[predicated_label] -= sample
