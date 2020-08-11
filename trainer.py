import os
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import models
import dataset
import utils
# import mc_dropout 

from alzheimers import utils as alzheimers_utils

tfd = tfp.distributions

def train(X, y, config):
	run_all_folds(X, y, train=True, config=config)

def evaluate(X, y, config):
	run_all_folds(X, y, train=False, config=config)

def run_all_folds(X, y, train, config):
	kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
	fold=1
	all_rmses = []
	all_nlls = []
	n_feature_sets = len(X)
	for train_index, test_index in kf.split(y):
		print('Fold {}'.format(fold))

		if config.dataset=='msd':
			train_index = [x for x in range(463715)]
			test_index = [x for x in range(463715, 515345)]

		y_train, y_val = y[train_index], y[test_index]
		x_train = [i[train_index] for i in X]
		x_val = [i[test_index] for i in X]
		if config.dataset in ['alzheimers_test']:
			alzheimers_test_data = dataset._alzheimers_test(config)
			x_val = [np.array(alzheimers_test_data['{}'.format(i)]) for i in range(n_feature_sets)]
			y_val = np.array(alzheimers_test_data['y'])
			print('Alzheimers Testing..')
			[print('Shape of feature set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(x_val)]

		if config.dataset in ['alzheimers', 'alzheimers_test']:
			assert x_train[-1].shape[-1] == 6373, 'not compare'
			x_train[-1], x_val[-1] = alzheimers_utils.normalize_compare_features(x_train[-1], x_val[-1])

		else:
			for i in range(n_feature_sets):
				x_train[i], x_val[i] = utils.standard_scale(x_train[i], x_val[i])

		rmse, nll = train_deep_ensemble(x_train, y_train, x_val, y_val, fold, config, train=train, verbose=config.verbose)
		all_rmses.append(rmse)
		all_nlls.append(nll)
		fold+=1
		print('='*20)

		if config.dataset in ['msd', 'alzheimers', 'alzheimers_test']:
			break

	print('Final {} fold results'.format(config.n_folds))
	print('val rmse {:.3f}, +/- {:.3f}'.format(np.mean(all_rmses), np.std(all_rmses)))
	[print('feature set {}, val nll {:.3f}, +/- {:.3f}'.format(i, np.mean(all_nlls, axis=0)[i], np.std(all_nlls, axis=0)[i]))
	 for i in range(n_feature_sets)]
	print(['{:.3f} {:.3f}'.format(np.mean(all_nlls, axis=0)[i], np.std(all_nlls, axis=0)[i]) 
		for i in range(n_feature_sets)])

def train_a_model(
	model_id, fold,
	x_train, y_train,
	x_val, y_val, config):

	model,_ = models.build_model(config)

	negloglik = lambda y, p_y: -p_y.log_prob(y)
	custom_mse = lambda y, p_y: tf.keras.losses.mean_squared_error(y, p_y.mean())
	# mse_wrapped = utils.MeanMetricWrapper(custom_mse, name='custom_mse')

	checkpoint_filepath = os.path.join(config.model_dir, 'fold_{}_nll_{}.h5'.format(fold, model_id))
	checkpointer = tf.keras.callbacks.ModelCheckpoint(
			checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
			save_weights_only=True, mode='auto', save_freq='epoch')

	if config.build_model == 'combined_pog':
		model.compile(optimizer=tf.optimizers.Adam(learning_rate=config.lr),
					  loss=[negloglik]*len(x_train))
		hist = model.fit(x_train, [y_train]*len(x_train),
						batch_size=config.batch_size,
						epochs=config.epochs,
						verbose=config.verbose,
						callbacks=[checkpointer],
						validation_data=(x_val, [y_val]*len(x_train)))

	elif config.build_model == 'combined_multivariate' or config.build_model == 'gaussian':
		model.compile(optimizer=tf.optimizers.Adam(learning_rate=config.lr),
					  loss=[negloglik])
		hist = model.fit(x_train, y_train,
						batch_size=config.batch_size,
						epochs=config.epochs,
						verbose=config.verbose,
						callbacks=[checkpointer],
						validation_data=(x_val, y_val))


	epoch_val_losses = hist.history['val_loss']
	best_epoch_val_loss, best_epoch = np.min(epoch_val_losses), np.argmin(epoch_val_losses)+1
	best_epoch_train_loss = hist.history['loss'][best_epoch-1]

	print('Model id: ', model_id)
	print('Best Epoch: {:d}'.format(best_epoch))
	print('Train NLL: {:.3f}'.format(best_epoch_train_loss)) 
	print('Val NLL: {:.3f}'.format(best_epoch_val_loss)) 

	model.load_weights(os.path.join(config.model_dir, 'fold_{}_nll_{}.h5'.format(fold, model_id)))

	return model, [best_epoch_train_loss, best_epoch_val_loss]

def train_deep_ensemble(x_train, y_train, x_val, y_val, fold, config, train=False, verbose=0):

	n_feature_sets = len(x_train)
	train_nlls, val_nlls = [], []
	mus = []
	featurewise_sigmas = [[] for i in range(n_feature_sets)]
	# ensemble_preds = []

	for model_id in range(config.n_models):

		if train:
			if config.build_model == 'gaussian' and config.mod_split != 'none':
				gaussian_split_models = []
				for i in range(config.n_feature_sets):
					new_model_id = str(model_id)+'_'+str(i)
					config.input_feature_length = config.feature_split_lengths[i]
					model, results = train_a_model(new_model_id, fold, x_train[i], y_train, x_val[i], y_val, config)
					gaussian_split_models.append(model)
			else:
				model, results = train_a_model(model_id, fold, x_train, y_train, x_val, y_val, config)
				train_nlls.append(results[0])
				val_nlls.append(results[1])
		else:
			if config.build_model == 'gaussian' and config.mod_split != 'none':
				gaussian_split_models = []
				for i in range(config.n_feature_sets):
					config.input_feature_length = config.feature_split_lengths[i]
					new_model_id = str(model_id)+'_'+str(i)
					model, _ = models.build_model(config)
					model.load_weights(os.path.join(config.model_dir, 'fold_{}_nll_{}.h5'.format(fold, new_model_id)))
					gaussian_split_models.append(model)
			else:
				model, _ = models.build_model(config)
				model.load_weights(os.path.join(config.model_dir, 'fold_{}_nll_{}.h5'.format(fold, model_id)))


		y_val = y_val.reshape(-1,1)

		if config.build_model == 'gaussian' and config.mod_split != 'none':
			gaussian_split_preds = []
			# print(x_val[0].shape)
			# exit()
			for i in range(config.n_feature_sets):
				gaussian_split_preds.append(gaussian_split_models[i](x_val[i]))
		else:
			preds = model(x_val)


		# ensemble_preds.append(preds)
		if config.build_model == 'gaussian' and config.mod_split != 'none':
			mu = [gaussian_split_preds[i].mean().numpy()[:,0] for i in range(config.n_feature_sets)]
			mu = np.sum(mu, axis=0) / config.n_feature_sets
			mus.append(mu)
		elif config.build_model == 'combined_multivariate' or config.build_model=='gaussian':
			mus.append(preds.mean().numpy()[:,0])
		elif config.build_model == 'combined_pog':
			mus.append(preds[0].mean().numpy())

		for i in range(n_feature_sets):
			if config.build_model == 'gaussian' and config.mod_split != 'none':
				featurewise_sigmas[i].append(gaussian_split_preds[i].stddev().numpy())
			elif config.build_model == 'combined_multivariate' or config.build_model == 'gaussian':
				featurewise_sigmas[i].append(preds.stddev().numpy()[:,i:i+1])
			elif config.build_model == 'combined_pog':
				featurewise_sigmas[i].append(preds[i].stddev().numpy())


		val_rmse = mean_squared_error(y_val,mus[model_id], squared=False)
		print('Val RMSE: {:.3f}'.format(val_rmse))

		n_val_samples = y_val.shape[0]
		if config.verbose > 1:
			for i in range(n_val_samples):
				stddev_print_string = ''
				for j in range(n_feature_sets):
					stddev_print_string += '\t\tStd Dev set {}: {:.5f}'.format(j, featurewise_sigmas[j][model_id][i][0])
				print('Pred: {:.3f}'.format(mus[model_id][i][0]), '\tTrue: {:.3f}'.format(y_val[i][0]), stddev_print_string)
		print('-'*20)

	# shape of mus - (5, 26)
	# shape of std - (3, 5, 26)

	if config.mixture_approximation == 'gaussian':
		ensemble_mus = np.mean(mus, axis=0).reshape(-1,1)
		ensemble_sigmas = []
		for i in range(n_feature_sets):
			ensemble_sigma = np.sqrt(np.mean(np.square(featurewise_sigmas[i]) + np.square(ensemble_mus), axis=0).reshape(-1,1) - np.square(ensemble_mus))
			ensemble_sigmas.append(ensemble_sigma)

		# ensemble_mus = np.squeeze(ensemble_mus, axis=-1)
		# ensemble_sigmas = np.squeeze(ensemble_sigmas, axis=-1)

		ensemble_val_nll = []
		for i in range(n_feature_sets):
			distributions = tfd.Normal(loc=ensemble_mus, scale=ensemble_sigmas[i])
			ensemble_val_nll.append(-1*np.mean(distributions.log_prob(y_val)))

	elif config.mixture_approximation == 'none':
		mix_prob = 1/config.n_models
		ensemble_normal = []
		ensemble_normal_model = tf.keras.models.Sequential([
				Input((config.n_models, 2)),
				tfp.layers.DistributionLambda(
					make_distribution_fn=lambda t: tfd.MixtureSameFamily(
						mixture_distribution=tfd.Categorical(
							probs=[mix_prob]*config.n_models),
						components_distribution=tfd.Normal(
							loc=t[...,0],       # One for each component.
							scale=t[...,1])))
			])
		mus = np.squeeze(mus, axis=-1)
		featurewise_sigmas = np.squeeze(featurewise_sigmas, axis=-1)

		for i in range(n_feature_sets):
			ensemble_normal.append(ensemble_normal_model(np.stack([np.array(mus).T,
			 np.array(featurewise_sigmas[i]).T], axis=-1)))

		ensemble_mus = ensemble_normal[0].mean().numpy()
		ensemble_sigmas = []
		for i in range(n_feature_sets):
			ensemble_sigmas.append(ensemble_normal[i].stddev().numpy())

		ensemble_val_nll = []
		for i in range(n_feature_sets):
			ensemble_val_nll.append(-1*np.mean(ensemble_normal[i].log_prob(y_val)))

		ensemble_mus = np.expand_dims(ensemble_mus, axis=-1)
		ensemble_sigmas = np.expand_dims(ensemble_sigmas, axis=-1)

	ensemble_val_rmse = mean_squared_error(y_val, ensemble_mus, squared=False)

	print('Deep Ensemble val rmse {:.3f}'.format(ensemble_val_rmse))
	print('Deep Ensemble val nll {}'.format(ensemble_val_nll))
	if verbose > 0:
		print('Deep Ensemble Results')
		for i in range(n_val_samples):
			stddev_print_string = ''
			for j in range(n_feature_sets):
				stddev_print_string += '\t\tStd Dev set {}: {:.5f}'.format(j, ensemble_sigmas[j][i][0])
			print('Pred: {:.3f}'.format(ensemble_mus[i][0]), '\tTrue: {:.3f}'.format(y_val[i][0]), stddev_print_string)

	return ensemble_val_rmse, ensemble_val_nll



