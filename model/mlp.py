import os
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np

from util.loss_functions import *
#from util.loss_functions import CrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys
import logging

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.1, epochs=50, weight_decay=0.02):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        # self.cost = cost

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        self.weight_decay=weight_decay
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        elif loss == 'ce':
            self.loss = CrossEntropyError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "tanh"
        self.layers.append(LogisticLayer(train.input.shape[1], 128, 
                           None, inputActivation, False))

        # Output layer
        # outputActivation = "softmax"
        self.layers.append(LogisticLayer(128, 10, 
                           None, outputActivation, True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        inputForNextLayer=inp
        for layer in self.layers:
            # if (not layer.isClassifierLayer):
            #     inputForNextLayer=layer.forward(inputForNextLayer)
            # else:
            #     inputForNextLayer=layer.forward(np.insert(inputForNextLayer, 0, 1))
            inputForNextLayer=layer.forward(inputForNextLayer)
            inputForNextLayer=np.insert(inputForNextLayer, 0, 1)
        
    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        # inputForNextLayer=inp
        # for layer in self.layers:
            # Compute the derivatives w.r.t to the error
            # Please note the treatment of nextDerivatives and nextWeights
            # in case of an output layer
            # layer.computeDerivative(self.loss.calculateDerivative(
            #                              label,self.layer.outp), 1.0)

            # Update weights in the online learning fashion
            # layer.updateWeights(self.learningRate)
        # return self.loss.calculateError(target,self._get_output_layer().outp)
        pass
    
    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
            layer.updateWeights(learningRate)
        pass
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # from util.loss_functions import DifferentError

        learned = False
        iteration = 0

        # Train for some epochs if the error is not 0
        perror=9999990
        while not learned:
            totalError = 0
            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):
                # print("dataset new")
                self._feed_forward(input)
                # if (self._get_output_layer().outp!=label):
                error = self.loss.calculateError(label, self._get_output_layer().outp)
                next_derivates=self.loss.calculateDerivative(label,self._get_output_layer().outp)
                next_weights=1.0
                if (error!=0):
                    for layer in reversed(self.layers):
                        # print("Layer index:")
                        # print(next_derivates)
                        # print(self.layers.index(layer))
                        if (layer.isClassifierLayer):
                            # print("set derivate output")
                            # # print(self.loss.calculateDerivative(label,layer.outp))
                            # layer.deltas=self.loss.calculateError(label,layer.outp)*layer.activationDerivative(layer.outp)
                            # print(layer.deltas.shape)
                            # next_derivates=layer.deltas#*layer.deltas
                            # layer.deltas=layer.deltas.T
                            # print(layer.outp.shape)
                            # next_derivates=layer.computeDerivativeOutput(next_derivates)
                            # layer.deltas=np.dot(next_derivates,layer.activationDerivative(layer.outp)).flatten()
                            layer.deltas=next_derivates#*layer.activationDerivative(layer.outp)).flatten()
                            # next_derivates=next_derivates.reshape(-1,1)
                            # layer.deltas=next_derivates*layer.weights#next_derivates*layer.activationDerivative(layer.outp)
                            # next_derivates=layer.computeDerivative(next_derivates,1.0)#np.ones(layer.outp.shape[0]))
                            # next_derivates=layer.computeDerivative(self.loss.calculateDerivative(label,layer.outp),1.0)#np.ones(layer.outp.shape[0]))
                        else:
                            # print("set derivate hidden")
                            # # print(next_derivates)
                            tmp=np.delete(next_weights,0,axis=0)
                            # tmp1=next_derivates.dot(tmp.T)
                            # # tmp=next_derivates.dot(np.delete(next_weights,0,axis=0).T)
                            # # print(tmp)
                            # layer.deltas=tmp1*layer.activationDerivative(layer.outp)
                            # next_derivates=layer.deltas
                            # layer.deltas=layer.deltas
                            next_derivates=layer.computeDerivative(next_derivates,tmp.T)
                            # # print(next_derivates.shape[0])
                        # print(layer.outp)
                        # next_derivates=layer.computeDerivative(next_derivates,np.asarray(next_weights).T)
                        layer.updateWeights(self.learningRate*self.weight_decay)
                        next_weights=np.asarray(layer.weights)
                        # print(next_weights)
                        
                        # print(next_weights.shape)
                        # print(next_derivates.shape)
                        
                    # print("update weight")
                    # self._update_weights(self.learningRate)
                    totalError += error

            iteration += 1
            
            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, -totalError)
            if verbose:
                accuracy = accuracy_score([np.argmax(x) for x in self.validationSet.label ],
                                          self.evaluate(self.validationSet))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")
            if totalError == 0 or iteration >= self.epochs or perror<totalError:
                # stop criteria is reached
                learned = True
            perror=totalError
        pass



    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        self._feed_forward(test_instance)
        outp=self._get_output_layer().outp
        if (self._get_output_layer().nOut==1):
            return outp >0.5
        else:
            return np.argmax(outp,axis=0)#np.argmax(target)

    def oneHot(self, pos,numClass=10):
        tmp=np.zeros(numClass)
        tmp[pos]=1.0
        return tmp       

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet#.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test.input))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
