import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import *

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import glob, os, math, time
import numpy as np
np.random.seed(0)
from pathlib import Path
import matplotlib.pyplot as plt

import load_dataset
# from ..utils import EasyDict

tfd = tfp.distributions

def standard_scale(x_train, x_test):
	scalar = StandardScaler()
	scalar.fit(x_train)
	x_train = scalar.transform(x_train)
	x_test = scalar.transform(x_test)
	return x_train, x_test

def create_feature_extractor_block(x):
	# x = Dense(16, activation='relu')(x)
	# x = BatchNormalization()(x)
	# x = Dense(8, activation='relu')(x)
	# x = BatchNormalization()(x)
	# x = Dropout(0.2)(x)

	x = Dense(50, activation='relu')(x)
	return x

def create_stddev_block(x):
	# x = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(x)
	x = Dense(1)(x)
	return x

def create_mu_block(x_list):
	x = Concatenate()(x_list)
	# x = Dense(8, activation='relu')(x)
	# x = BatchNormalization()(x)
	# x = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(x)
	x = Dense(1)(x)
	return x

def create_gaussian_output(mu, stddev, name):
	x = Concatenate()([mu, stddev])
	x = tfp.layers.DistributionLambda(
		lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6), name=name)(x)
	return x

def create_combined_model(feature_split_lengths):
	'''
	feature splits is a list of n features per splits
	'''
	n_feature_sets = len(feature_split_lengths)

	inputs = []
	for i in range(n_feature_sets):
		inputs.append(Input((feature_split_lengths[i],)))

	feature_extractors = []
	for i in range(n_feature_sets):
		feature_extractors.append(create_feature_extractor_block(inputs[i]))

	stddevs = []
	for i in range(n_feature_sets):
		stddevs.append(create_stddev_block(feature_extractors[i]))

	mu = create_mu_block(feature_extractors)

	outputs = []
	for i in range(n_feature_sets):
		outputs.append(create_gaussian_output(mu, stddevs[i], name='set_{}'.format(i)))

	return tf.keras.models.Model(inputs=inputs, outputs=outputs)

def train_a_model(
	model_id, model_dir, fold,
	x_train, y_train,
	x_val, y_val):

	feature_split_lengths = [i.shape[1] for i in x_train]
	model = create_combined_model(feature_split_lengths)
	# lr = 0.01
	# epochs = 2000	
	lr = 0.1
	epochs = 30


	negloglik = lambda y, p_y: -p_y.log_prob(y)
	custom_mse = lambda y, p_y: tf.keras.losses.mean_squared_error(y, p_y.mean())
	# mse_wrapped = utils.MeanMetricWrapper(custom_mse, name='custom_mse')

	checkpoint_filepath = os.path.join(model_dir, 'fold_{}_nll_{}.h5'.format(fold, model_id))
	checkpointer = tf.keras.callbacks.ModelCheckpoint(
			checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
			save_weights_only=True, mode='auto', save_freq='epoch')


	model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
				  loss=[negloglik]*len(x_train))
				  # metrics=[mse_wrapped, mse_wrapped, mse_wrapped])

	hist = model.fit(x_train, [y_train]*len(x_train),
					batch_size=100,
					epochs=epochs,
					verbose=0,
					callbacks=[checkpointer],
					validation_data=(x_val, [y_val]*len(x_train)))

	epoch_val_losses = hist.history['val_loss']
	best_epoch_val_loss, best_epoch = np.min(epoch_val_losses), np.argmin(epoch_val_losses)+1
	best_epoch_train_loss = hist.history['loss'][best_epoch-1]

	print('Model id: ', model_id)
	print('Best Epoch: {:d}'.format(best_epoch))
	print('Train NLL: {:.3f}'.format(best_epoch_train_loss)) 
	print('Val NLL: {:.3f}'.format(best_epoch_val_loss)) 

	model.load_weights(os.path.join(model_dir, 'fold_{}_nll_{}.h5'.format(fold, model_id)))

	return model, [best_epoch_train_loss, best_epoch_val_loss]

def train_deep_ensemble(x_train, y_train, x_val, y_val, fold, config, train=False, verbose=0):

	n_feature_sets = len(x_train)
	train_nlls, val_nlls = [], []
	mus = []
	featurewise_sigmas = [[] for i in range(n_feature_sets)]
	ensemble_preds = []

	for model_id in range(config.n_models):

		if train:
			model, results = train_a_model(model_id, config.model_dir, fold, x_train, y_train, x_val, y_val)
			train_nlls.append(results[0])
			val_nlls.append(results[1])
		else:
			model = create_combined_model([i.shape[1] for i in X])
			model.load_weights(os.path.join(model_dir, 'fold_{}_nll_{}.h5'.format(fold, model_id)))

		y_val = y_val.reshape(-1,1)
		preds = model(x_val)

		ensemble_preds.append(preds)
		mus.append(preds[0].mean().numpy())
		for i in range(n_feature_sets):
			featurewise_sigmas[i].append(preds[i].stddev().numpy())

		val_rmse = mean_squared_error(y_val, mus[model_id], squared=False)
		print('Val RMSE: {:.3f}'.format(val_rmse))

		n_val_samples = y_val.shape[0]
		if verbose == 1:
			for i in range(n_val_samples):
				stddev_print_string = ''
				for j in range(n_feature_sets):
					stddev_print_string += '\t\tStd Dev set {}: {:.5f}'.format(j, featurewise_sigmas[j][model_id][i][0])
				print('Pred: {:.3f}'.format(mus[model_id][i][0]), '\tTrue: {:.3f}'.format(y_val[i][0]), stddev_print_string)
		print('-'*20)

	ensemble_mus = np.mean(mus, axis=0).reshape(-1,1)
	ensemble_sigmas = []
	for i in range(n_feature_sets):
		ensemble_sigma = np.sqrt(np.mean(np.square(featurewise_sigmas[i]) + np.square(mus), axis=0).reshape(-1,1) - np.square(ensemble_mus))
		ensemble_sigmas.append(ensemble_sigma)
	
	ensemble_val_rmse = mean_squared_error(y_val, ensemble_mus, squared=False)

	print('Deep Ensemble val rmse {:.3f}'.format(ensemble_val_rmse))
	if verbose == 1:
		print('Deep Ensemble Results')
		for i in range(n_val_samples):
			stddev_print_string = ''
			for j in range(n_feature_sets):
				stddev_print_string += '\tStd Dev set {}: {:.5f}'.format(j, ensemble_sigmas[j][i][0])
			print('Pred: {:.3f}'.format(ensemble_mus[i][0]), '\tTrue: {:.3f}'.format(y_val[i][0]), stddev_print_string)

	ensemble_val_nll = []
	for i in range(n_feature_sets):
		distributions = tfd.Normal(loc=ensemble_mus, scale=ensemble_sigmas[i])
		ensemble_val_nll.append(-1*np.mean(distributions.log_prob(y_val)))
	return ensemble_val_rmse, ensemble_val_nll

# Config to choose the hyperparameters for everything
class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name] 

config = EasyDict({
	'n_models' :  5,
	'model_dir' : 'deepmind',
	# 'model_dir' : 'temp',
	'dataset': 'boston',
	'feature_selection': [],
	'test_split': 0.05,
	'mod_split': 'computation_split',
	'n_folds': 20,
	'train': False
	})

data = load_dataset.load_dataset(config)
n_feature_sets = len(data.keys()) - 1
X = [np.array(data['{}'.format(i)]) for i in range(n_feature_sets)]
y = np.array(data['y'])
n_samples = len(y)

print('Number of feature sets ', n_feature_sets)
[print('Shape of feature set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(X)]

model_dir = Path(config.model_dir)
model_dir.mkdir(parents=True, exist_ok=True)

kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
fold=1
all_rmses = []
all_nlls = []
for train_index, test_index in kf.split(y):
	print('Fold {}'.format(fold))
	y_train, y_val = y[train_index], y[test_index]
	x_train = [i[train_index] for i in X]
	x_val = [i[test_index] for i in X]
	for i in range(n_feature_sets):
		x_train[i], x_val[i] = standard_scale(x_train[i], x_val[i])
	rmse, nll = train_deep_ensemble(x_train, y_train, x_val, y_val, fold, config, train=config.train)
	all_rmses.append(rmse)
	all_nlls.append(nll)
	fold+=1
	print('='*20)

print(np.array(all_nlls).shape)
print('Final {} fold results'.format(config.n_folds))
print('val rmse {:.3f}, +/- {:.3f}'.format(np.mean(all_rmses), np.std(all_rmses)))
[print('feature set {}, val nll {:.3f}, +/- {:.3f}'.format(i, np.mean(all_nlls, axis=0)[i], 
	np.std(all_nlls, axis=0)[i])) for i in range(n_feature_sets)]
exit()
## DEFER CALIBRATION PLOT ##

def defer_analysis(true_values, ensemble_preds, defer_based_on):
	true_values = np.squeeze(true_values, axis=-1)
	ensemble_preds = np.squeeze(ensemble_preds, axis=-1)
	defer_based_on = np.squeeze(defer_based_on, axis=-1)
	defered_rmse_list, non_defered_rmse_list = [], []
	for i in range(ensemble_preds.shape[0]+1):
		print('\n{} datapoints deferred'.format(i))

		if i==ensemble_preds.shape[0]:
			defered_rmse = mean_squared_error(true_values, ensemble_preds, squared=False)
		elif i==0:
			defered_rmse = 0
		else:
			print(true_values[np.argsort(defer_based_on)][-10:].shape)
			defered_rmse = mean_squared_error(
				true_values[np.argsort(defer_based_on)][-i:], 
				ensemble_preds[np.argsort(defer_based_on)][-i:], squared=False)
		defered_rmse_list.append(defered_rmse)

		if i==0:
			non_defered_rmse = mean_squared_error(true_values, ensemble_preds, squared=False)
		elif i==ensemble_preds.shape[0]:
			non_defered_rmse = 0
		else:
			non_defered_rmse = mean_squared_error(
				true_values[np.argsort(defer_based_on)][:-i], 
				ensemble_preds[np.argsort(defer_based_on)][:-i], squared=False)

		non_defered_rmse_list.append(non_defered_rmse)
		
		print('Defered RMSE : {:.3f}'.format(defered_rmse))
		print('Not Defered RMSE : {:.3f}'.format(non_defered_rmse))
	return defered_rmse_list, non_defered_rmse_list

for i in range(n_feature_sets):
	print('feature set {}'.format(i))
	defered_rmse_list, non_defered_rmse_list = defer_value(y_val, ensemble_mus, ensemble_sigmas[i] )

	plt.subplot(n_feature_sets, 1, i)
	plt.plot(range(ensemble_mus.shape[0]+1), defered_rmse_list, label='Defered RMSE')
	plt.plot(range(ensemble_mus.shape[0]+1), non_defered_rmse_list, label='Non Defered RMSE')
	plt.legend()
	plt.xlabel('No. of datapoints defered')
	plt.xticks(range(ensemble_mus.shape[0]+1))
	plt.yticks(range(0,5))
	plt.title('feature set {}'.format(i))
	plt.grid()

plt.savefig('combined_ensemble_calibration.png')
plt.show()
