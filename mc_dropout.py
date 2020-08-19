# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his 
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".
# Edited by Valyo Yolovski

import warnings
warnings.filterwarnings("ignore")

import math
import os
from scipy.special import logsumexp
import numpy as np
from scipy import stats
from sklearn.preprocessing import scale

#get_ipython().run_line_magic('matplotlib', 'notebook')
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import tensorflow.keras.constraints as constraints

import time


class net:

    def __init__(self, X_train, y_train, n_hidden, n_epochs = 40, weight_decay=0.0001, lengthscale=0.01,
        normalize = False, x_val = None, y_val = None, verbose = 0, train = True, model_dir = 'models',
        model_id = 0, fold = 0):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[ self.std_X_train == 0 ] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[ 1 ])
            self.mean_X_train = np.zeros(X_train.shape[ 1 ])


        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / np.full(X_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)
        
        # Normalize training data
        
        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        y_train_normalized = np.array(y_train_normalized, ndmin = 2).T

        # Normalize val data
        
        y_val_normalized = (y_val - self.mean_y_train) / self.std_y_train
        y_val_normalized = np.array(y_val_normalized, ndmin = 2).T

        N = X_train.shape[0]
        dropout = 0.05
        batch_size = 128
        lengthscale = 1e-2
        tau = lengthscale**2 * (1 - dropout) / (2 * N * weight_decay) # We can calculate tau here
        reg = lengthscale**2 * (1 - dropout) / (2. * N * tau) # Regularization caluclations

        # Create a neural network as specified 
        
        model = Sequential()
        model.add(Dropout(dropout, input_shape=(X_train.shape[1],)))
        model.add(Dense(n_hidden[0], activation='relu', kernel_regularizer=l2(reg)))
        for i in range(len(n_hidden) - 1):
            model.add(Dropout(dropout))
            model.add(Dense(n_hidden[i+1], activation='relu', kernel_regularizer=l2(reg)))
        model.add(Dropout(dropout))
        model.add(Dense(y_train_normalized.shape[1], kernel_regularizer=l2(reg)))

        model.compile(loss='mean_squared_error', optimizer='adam')

        # We iterate the learning process
        if train:
            checkpoint_filepath = os.path.join(model_dir, 'fold_{}_nll_{}.h5'.format(fold, model_id))
            checkpointer = tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
                    save_weights_only=True, mode='auto', save_freq='epoch')

            start_time = time.time()
            hist = model.fit(X_train, y_train_normalized, batch_size=batch_size, epochs=n_epochs, verbose=verbose, 
                callbacks=[checkpointer], validation_data=(x_val, y_val_normalized))
            self.model = model
            self.tau = tau
            self.running_time = time.time() - start_time
            print('Best epoch' , np.argmin(hist.history['val_loss']))
        else:
            model.load_weights(os.path.join(model_dir, 'fold_{}_nll_{}.h5'.format(fold, model_id)))
            self.model = model
            self.tau = tau

        
    def predict(self, X_test, y_test):

        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data
            
            @param y_test   The array with target values for the test data
    
            @return rmse    RMSE on the test data when using MC Dropout
            
            @return y_pred  1000 predictions for each target from the test data
            
            @return m       The predictive mean for the test target variables.
            
            @return v       The predictive variance for the test target
                            variables.

        """

        X_test = np.array(X_test, ndmin = 2)
        y_test = np.array(y_test, ndmin = 2).T

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        model = self.model
        standard_pred = model.predict(X_test, batch_size=100, verbose=0) # standard predictions
        standard_pred = standard_pred * self.std_y_train + self.mean_y_train #unnormalizing data
        rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze())**2.)**0.5 # and standard rmse calculations.
        
        # Let's make the MC dropout predictions. Here we get the uncertainty.
        # Let's choose how many times we predict a single point (how many stochastic passes over the network)
        
        # T = 1000
        T = 500
        
        # Here we make a function that uses dropout in test time.
        # predict_stochastic = partial(model)

        # Here in Yt_hat we store the results for each pass
        Yt_hat = np.array([model([X_test, 1], training=True) for _ in range(T)])
        
        #Un-normalize 
        Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
        
        #Calculate the predictive mean
        MC_pred = np.mean(Yt_hat, 0)

        # Calculate the predictive variance
        predictive_variance = np.var(Yt_hat, axis=0)
        
        # The **MC** RMSE
        rmse = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5

        # We compute the test log-likelihood
        ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat)**2., 0) - np.log(T) 
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        test_ll = np.mean(ll)
        
        # print('Standard rmse %f' % (rmse_standard_pred))
        # print('MC rmse %f' % (rmse))
        # print('test_ll %f' % (test_ll))

        # return rmse, Yt_hat, MC_pred, predictive_variance
        return rmse, -1*test_ll
