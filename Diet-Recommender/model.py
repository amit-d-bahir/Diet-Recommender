import scipy.io
import pandas as pd
import numpy as np
import random
import copy
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Creating and Training a multi-layer perceptron neural networks using Levenberg-Marquardt(LM) training
class NN:
    def __init__(self, nn):
        """
        :param nn: Structure of the NN
        Example: [2, 3, 4, 1] represents a NN with
                two inputs
                two hidden layers with 3 and 4, respectively
                one linear output layer
        """
        self.net = {'nn': nn, 'M': len(nn) - 1, 'layers': nn[1:]}
        self.net['w'], self.net['b'], self.net['N'] = self.create_w()

    # Initializing weights
    def create_w(self):
        w = []
        b = []
        weights = 0
        for i in range(self.net['M']):
            w.append(np.random.rand(self.net['nn'][i+1], self.net['nn'][i]) - 0.5)
            b.append(np.random.rand(self.net['nn'][i+1]))
            weights += self.net['nn'][i+1] * (self.net['nn'][i] + 1)

        return w, b, weights
    
    # Sigmoid function
    @staticmethod
    def sigmoid(x):
      x = np.where(np.isfinite(x), x, 0)
      x = np.where(np.isnan(x), x, 0)
      return 1.0/(1 + np.exp(-x))
    
    # Forward Propagation
    def feed_forward(self, x):
        ret = x
        #print("----Fwd Prop----")
        for i in range(len(self.net['w'])):
            ret = np.dot(ret, self.net['w'][i].transpose())
            for j in range(ret.shape[0]):
                ret[j][:] += self.net['b'][i]
            ret = NN.sigmoid(ret)
        return ret
    
    # Mapping weights for each layer
    @staticmethod
    def map_weights(w, b):
        layer_to_w = {}
        w_to_layer = {}
        cnt = 0
        for layer in range(len(w)):
            for i in range(w[layer].shape[0]):
                layer_to_w[(layer, i, -1)] = cnt
                w_to_layer[cnt] = (layer, i, -1)
                cnt += 1
                for j in range(w[layer].shape[1]):
                    layer_to_w[(layer, i, j)] = cnt
                    w_to_layer[cnt] = (layer, i, j)
                    cnt += 1
        return layer_to_w, w_to_layer
    
    #Training the ANN using LM training method
    def train_lm(self, x, y, min_error=0.01, epochs=50, u=0.01):
        patterns = len(x)
        output_neurons = self.net['nn'][len(self.net['nn']) - 1]
        layer_to_w, w_to_layer = NN.map_weights(self.net['w'], self.net['b'])
        prev_error = 100
        m = 0
        for epoch in range(epochs):
            print("Epoch: " + str(epoch))
            prediction = self.feed_forward(x)
            e = NN.create_e(patterns, output_neurons, y, prediction)
            error = 0.5 * np.sum(np.square(e))
            if error <= min_error:
                break
            j = self.create_j(x, y, e, self.net['N'], layer_to_w, w_to_layer, patterns, output_neurons)
            tmp = np.dot(np.linalg.pinv(np.dot(j.transpose(), j) + u*np.identity(j.shape[1])), np.dot(j.transpose(), e.transpose()))

            tmp_w = self.net['w']
            tmp_b = self.net['b']
            #print("----Before tmp for loop----")
            for i in range(len(tmp)):
                tup = w_to_layer[i]
                if tup[2] == -1:
                    self.net['b'][tup[0]][tup[1]] -= tmp[i]
                else:
                    self.net['w'][tup[0]][tup[1]][tup[2]] -= tmp[i]

            #print("----After tmp for loop----")
            e = NN.create_e(patterns, output_neurons, y, prediction)
            error = 0.5 * np.sum(np.square(e))
            if error >= prev_error:
                u *= 10
                if m <= 5:
                    self.net['w'] = tmp_w
                    self.net['b'] = tmp_b
                    m += 1
                    continue
            else:
                m = 0
                #u /= 10
            prev_error = error

        return self.net['w'], self.net['b'], prev_error
    
    # For calculating error between the perdicted value and the true value
    @staticmethod
    def create_e(patterns, output_neurons, y, prediction):
        #print("----Create E----")
        e = np.zeros([1, patterns*output_neurons])
        cnt = 0
        for i in range(patterns):
            for j in range(output_neurons):
                e[0][cnt] = y[i][j] - prediction[i][j]
                cnt += 1

        return e

    # Used for Backward Propagation
    def create_j(self, x, y, e, weights, layer_to_w, w_to_layer, patterns, output_neurons):
        delta_w = 0.001
        #print("----Create J----")
        j = np.zeros([e.shape[1], weights])
        original_e = NN.create_e(patterns, output_neurons, y, self.feed_forward(x)).transpose()

        for i in range(weights):
            tup = w_to_layer[i]
            if tup[2] == -1:
                self.net['b'][tup[0]][tup[1]] += delta_w
            else:
                self.net['w'][tup[0]][tup[1]][tup[2]] += delta_w

            predict_e = NN.create_e(patterns, output_neurons, y, self.feed_forward(x)).transpose()
            subt = np.subtract(predict_e, original_e)
            for row in range(j.shape[0]):
                j[row][i] += subt[row]

            if tup[2] == -1:
                self.net['b'][tup[0]][tup[1]] -= delta_w
            else:
                self.net['w'][tup[0]][tup[1]][tup[2]] -= delta_w
        return j
    
    # Prediction function
    def predict(self, x):
        predict = self.feed_forward(x)
        #return predict
        for i in range(predict.shape[0]):
            for j in range(predict.shape[1]):
                if predict[i][j] >= 0.5:
                    predict[i][j] = 1
                else:
                    predict[i][j] = 0
        return predict