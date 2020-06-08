import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import numpy as np
np.random.seed(0)

import trainer
import life_trainer
import experiments
import utils
from opts import Opts

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import models
import dataset
import utils
from alzheimers import utils as alzheimers_utils

tfd = tfp.distributions

def train(X, y, config):
	run(X, y, train=True, config=config)

def evaluate(X, y, config):
	run(X, y, train=False, config=config)

def run(X, y, train, config):
	kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
	fold=1
	all_rmses = []
	all_nlls = []
	n_feature_sets = len(X)
	for train_index, test_index in kf.split(y):
		print('Fold {}'.format(fold))

		if config.dataset=='life_test':
			train_index = [x for x in range(len(y))]
			test_index = [x for x in range(len(y))]

		y_train, y_val = y[train_index], y[test_index]
		x_train = [i[train_index] for i in X]
		x_val = [i[test_index] for i in X]
		
		preds_train_all, preds_val_all = [0]*len(y_train), [0]*len(y_val)

		for i in range(n_feature_sets):
			x_train[i], x_val[i] = utils.standard_scale(x_train[i], x_val[i])

			model, loss = models.build_model(config)
			checkpoint_filepath = os.path.join(config.model_dir, 'fold_{}_feature_{}.h5'.format(fold, i))
			checkpointer = tf.keras.callbacks.ModelCheckpoint(
					checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
					save_weights_only=False, mode='auto', save_freq='epoch')
			if train:
				model.compile(optimizer=tf.optimizers.Adam(learning_rate=config.lr), loss=loss)

				hist = model.fit(x_train[i], y_train,
								batch_size=config.batch_size,
								epochs=config.epochs,
								verbose=config.verbose,
								callbacks=[checkpointer],
								validation_data=(x_val[i], y_val))
				epoch_val_losses = hist.history['val_loss']
				best_epoch_val_loss, best_epoch = np.min(epoch_val_losses), np.argmin(epoch_val_losses)+1
				best_epoch_train_loss = hist.history['loss'][best_epoch-1]
				print('Best Epoch: {:d}'.format(best_epoch))
			model = load_model(checkpoint_filepath)

			preds_train = model.predict(x_train[i])
			preds_val = model.predict(x_val[i])

			preds_train_all += preds_train[:, 0]
			preds_val_all += preds_val[:, 0]

			y_train = y_train.reshape(-1,1)
			y_val = y_val.reshape(-1,1)

			rmse_train = mean_squared_error(y_train, preds_train, squared=False)
			rmse_val = mean_squared_error(y_val, preds_val, squared=False)

			print("Feature {}, rmse train {}, rmse val {}".format(i, rmse_train, rmse_val))
		preds_train_all/=n_feature_sets
		preds_val_all/=n_feature_sets 
		print("Final train rmse {}, val rmse {}".format(mean_squared_error(preds_train_all, y_train, squared=False), mean_squared_error(preds_val_all, y_val, squared=False)))
		break


opts = Opts()
config = opts.parse()

data = dataset.load_dataset(config)

n_feature_sets = len(data.keys()) - 1
X = [np.array(data['{}'.format(i)]) for i in range(n_feature_sets)]
y = np.array(data['y'])

config.n_feature_sets = n_feature_sets
config.feature_split_lengths = [i.shape[1] for i in X]

print('Dataset used ', config.dataset)
print('Number of feature sets ', n_feature_sets)
[print('Shape of feature set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(X)]

utils.make_model_dir(config.model_dir)

if config.dataset in ['life']:
	config.units = 50
	config.n_folds = 10
	if config.task=='train':
		life_trainer.train(X, y, config)
	else:
		life_trainer.evaluate(X, y, config)
elif config.dataset in ['life_test']:
	life_trainer.evaluate(X, y, config)

#train
# python extras/life_trainer.py train --datasets_dir datasets --dataset life --model_dir life_hc --n_models 1 --epochs 1000 --build_model point --units_type absolute --mixture_approximation none

# evaluate n the missing features
# python extras/life_trainer.py evaluate --datasets_dir datasets --dataset life_test --model_dir life_hc --n_models 1  --build_model point --units_type absolute --mixture_approximation none
