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
import mc_dropout 

# from alzheimers import alz_utils as alzheimers_utils

from anc_ens import anc_ens, hyperparameters, utils as anc_utils, DataGen

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
	all_clusterwise_rmses = []
	n_feature_sets = len(X)

	# trying to replicate results from their paper
	if config.build_model=="anc_ens":
		for i in range(n_feature_sets):
			X[i], _ = utils.standard_scale(X[i], X[i])
		scale_c = np.std(y)
		shift_m = np.mean(y)
		y, _ = utils.standard_scale(y.reshape(-1, 1), y.reshape(-1, 1))
		print("shift_m {}, scale_c {}".format(shift_m, scale_c))

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

		# else:
		# 	for i in range(n_feature_sets):
		# 		x_train[i], x_val[i] = utils.standard_scale(x_train[i], x_val[i])

		if config.build_model == 'gaussian' and config.mod_split != 'none':
			rmse, nll, cluster_rmse = train_deep_ensemble(x_train, y_train, x_val, y_val, fold, config, train=train, verbose=config.verbose)
			all_clusterwise_rmses.append(cluster_rmse)
		elif config.build_model == 'anc_ens':
			rmse, nll, cluster_rmse = train_anchor_ensemble(x_train, y_train, x_val, y_val, fold, scale_c, shift_m, config)
			# train_anchor_ensemble(x_train, y_train, x_val, y_val, fold, config)
			all_clusterwise_rmses.append(cluster_rmse)
			# fold+=1
			# continue
		else:
			rmse, nll = train_deep_ensemble(x_train, y_train, x_val, y_val, fold, config, train=train, verbose=config.verbose)
		all_rmses.append(rmse)
		all_nlls.append(nll)
		fold+=1
		print('='*20)
		if config.dataset in ['msd', 'alzheimers', 'alzheimers_test']:
			break

	print('Final {} fold results'.format(config.n_folds))
	print('val rmse {:.3f}, +/- {:.3f}'.format(np.mean(all_rmses), np.std(all_rmses)))
	[print('feature set {}, val nll {:.3f}, +/- {:.3f}'.format(i, np.mean(all_nlls, axis=0)[i], np.std(all_nlls, axis=0)[i])) for i in range(n_feature_sets)]
	print(['{:.3f} {:.3f}'.format(np.mean(all_nlls, axis=0)[i], np.std(all_nlls, axis=0)[i]) for i in range(n_feature_sets)])

	if config.build_model == 'gaussian' and config.mod_split != 'none':
		[print('feature set {}, val rmse {:.3f}, +/- {:.3f}'.format(i, np.mean(all_clusterwise_rmses, axis=0)[i], np.std(all_clusterwise_rmses, axis=0)[i]))
	 for i in range(n_feature_sets)]

	if config.build_model == 'anc_ens':
		[print('feature set {}, val rmse {:.3f}, +/- {:.3f}'.format(i, np.mean(all_clusterwise_rmses, axis=0)[i], np.std(all_clusterwise_rmses, axis=0)[i]))
	 for i in range(n_feature_sets)]

def train_a_model(
	model_id, fold,
	x_train, y_train,
	x_val, y_val, config):

	if config.build_model == 'mc_dropout':
		model = mc_dropout.net(np.array(x_train), y_train, n_epochs=config.epochs, n_hidden=[50,25], 
			normalize=False, verbose=config.verbose, model_dir=config.model_dir, fold=fold, model_id=model_id,
			x_val=x_val, y_val=y_val)
		mc_rmse, nll = model.predict(np.array(x_train[0]), y_train)
		return model, [nll, nll]	
	else:

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
	gaussian_split_mus = []
	mc_dropout_rmses, mc_dropout_nlls = [], []
	gaussian_sigmas = []
	ensemble_clusterwise_val_rmse = []

	# ensemble_preds = []

	for model_id in range(config.n_models):

		if train:
			if config.build_model == 'mc_dropout' or (config.build_model == 'gaussian' and config.mod_split != 'none'):
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
			elif config.build_model == 'mc_dropout':
				gaussian_split_models = []
				for i in range(config.n_feature_sets):
					config.input_feature_length = config.feature_split_lengths[i]
					new_model_id = str(model_id)+'_'+str(i)
					model = mc_dropout.net(np.array(x_train[i]), y_train, n_epochs=config.epochs, n_hidden=[24,12], 
						normalize=False, verbose=config.verbose, train=False, model_dir=config.model_dir, fold=fold, model_id=model_id)
					gaussian_split_models.append(model)
				
			else:
				model, _ = models.build_model(config)
				model.load_weights(os.path.join(config.model_dir, 'fold_{}_nll_{}.h5'.format(fold, model_id)))


		y_val = y_val.reshape(-1,1)

		if config.build_model == 'gaussian' and config.mod_split != 'none':
			gaussian_split_preds = []
			for i in range(config.n_feature_sets):
				gaussian_split_preds.append(gaussian_split_models[i](x_val[i]))
		elif config.build_model == 'mc_dropout':
			for i in range(config.n_feature_sets):
				mc_rmse, nll = gaussian_split_models[i].predict(np.array(x_train[i]), y_train)
				print('Fold {}'.format(fold))
				print('Cluster {} RMSE: {:.5f}'.format(i, mc_rmse))
				print('Cluster {} NLL: {:.5f}'.format(i, nll))
				mc_dropout_rmses.append(mc_rmse)
				mc_dropout_nlls.append(nll)
			print('-'*20)
			continue 
		else:
			preds = model(x_val)

		# Get mus from models

		if config.build_model == 'gaussian' and config.mod_split != 'none':
			mu = [gaussian_split_preds[i].mean().numpy()[:,0] for i in range(config.n_feature_sets)]
			gaussian_split_mus.append(mu)
			mu = np.sum(mu, axis=0) / config.n_feature_sets
			mus.append(mu)
		elif config.build_model == 'combined_multivariate' or config.build_model=='gaussian':
			mus.append(preds.mean().numpy()[:,0])
		elif config.build_model == 'combined_pog':
			mus.append(preds[0].mean().numpy())

		# Get sigmas from models

		for i in range(n_feature_sets):
			if config.build_model == 'gaussian' and config.mod_split != 'none':
				featurewise_sigmas[i].append(gaussian_split_preds[i].stddev().numpy())
			elif config.build_model == 'combined_multivariate' or config.build_model == 'gaussian':
				featurewise_sigmas[i].append(preds.stddev().numpy()[:,i:i+1])
			elif config.build_model == 'combined_pog':
				featurewise_sigmas[i].append(preds[i].stddev().numpy())

		if config.build_model == 'gaussian' and config.mod_split == 'none':
			gaussian_sigmas.append(preds.stddev().numpy())

		# print results of models

		if config.build_model == 'gaussian' and config.mod_split != 'none':
			for i in range(config.n_feature_sets):
				print('Cluster {} RMSE: {:.3f}'.format(i, mean_squared_error(y_val,gaussian_split_mus[model_id][i], squared=False)))

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

	# not a deep ensemble
	if config.build_model == 'mc_dropout':
		return np.mean(mc_dropout_rmses), np.mean(mc_dropout_nlls)

	# shape of mus - (5, 26)
	# shape of std - (3, 5, 26)

	print("Shape of featurewise sigmas {}".format(np.asarray(featurewise_sigmas).shape))
	if config.mixture_approximation == 'gaussian' and config.mod_split!='none':
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

	# elif config.mixture_approximation == 'none':
	# 	mix_prob = 1/config.n_models
	# 	ensemble_normal = []
	# 	ensemble_normal_model = tf.keras.models.Sequential([
	# 			Input((config.n_models, 2)),
	# 			tfp.layers.DistributionLambda(
	# 				make_distribution_fn=lambda t: tfd.MixtureSameFamily(
	# 					mixture_distribution=tfd.Categorical(
	# 						probs=[mix_prob]*config.n_models),
	# 					components_distribution=tfd.Normal(
	# 						loc=t[...,0],       # One for each component.
	# 						scale=t[...,1])))
	# 		])
	# 	mus = np.squeeze(mus, axis=-1)
	# 	featurewise_sigmas = np.squeeze(featurewise_sigmas, axis=-1)

	# 	for i in range(n_feature_sets):
	# 		ensemble_normal.append(ensemble_normal_model(np.stack([np.array(mus).T,
	# 		 np.array(featurewise_sigmas[i]).T], axis=-1)))

	# 	ensemble_mus = ensemble_normal[0].mean().numpy()
	# 	ensemble_sigmas = []
	# 	for i in range(n_feature_sets):
	# 		ensemble_sigmas.append(ensemble_normal[i].stddev().numpy())

	# 	ensemble_val_nll = []
	# 	for i in range(n_feature_sets):
	# 		ensemble_val_nll.append(-1*np.mean(ensemble_normal[i].log_prob(y_val)))

	# 	ensemble_mus = np.expand_dims(ensemble_mus, axis=-1)
	# 	ensemble_sigmas = np.expand_dims(ensemble_sigmas, axis=-1)

	elif config.mod_split == 'none':
		ensemble_mus = np.mean(mus, axis=0).reshape(-1,1)
		ensemble_sigmas = []

		ensemble_sigma = np.sqrt(np.mean(np.square(gaussian_sigmas)+ np.square(ensemble_mus), axis=0).reshape(-1,1) - np.square(ensemble_mus))
		ensemble_sigmas.append(ensemble_sigma)

		# ensemble_mus = np.squeeze(ensemble_mus, axis=-1)
		# ensemble_sigmas = np.squeeze(ensemble_sigmas, axis=-1)

		ensemble_val_nll = []
		for i in range(n_feature_sets):
			distributions = tfd.Normal(loc=ensemble_mus, scale=ensemble_sigmas)
			ensemble_val_nll.append(-1*np.mean(distributions.log_prob(y_val)))

	if config.build_model == 'gaussian' and config.mod_split != 'none':
		for i in range(config.n_feature_sets):
			gaussian_split_ensemble_mus = np.mean(np.array(gaussian_split_mus)[:,i,:], axis=0)
			cluster_rmse = mean_squared_error(y_val, gaussian_split_ensemble_mus, squared=False)
			print('Deep Ensemble val rmse clusterwise {} RMSE: {:.3f}'.format(i, cluster_rmse ))
			ensemble_clusterwise_val_rmse.append(cluster_rmse)

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

	if config.build_model == 'gaussian' and config.mod_split != 'none':
		return ensemble_val_rmse, ensemble_val_nll, ensemble_clusterwise_val_rmse


	return ensemble_val_rmse, ensemble_val_nll


def train_anchor_ensemble(x_train, y_train, x_val, y_val, fold, scale_c, shift_m, config):
	is_print = False
	if(config.verbose):
		is_print = True

	hyp = hyperparameters.get_hyperparams(config.dataset, config.units)
	model_name = os.path.join(config.model_dir, '_ancens_fold_{}'.format(fold))

	all_features_ensemble_preds = []
	n_feature_sets = len(x_train)

	featurewise_nll = []

	featurewise_sigmas = [[] for i in range(n_feature_sets)]
	
	ensemble_clusterwise_val_rmse = []
	ensemble_sigmas = []
	
	for i in range(n_feature_sets):
		print("Feature set {}".format(i))
		
		ens = anc_ens.NN_ens(activation_fn='relu', 
							data_noise=hyp['data_noise'],
							b_0_var=hyp['b_0_var'], w_0_var=hyp['w_0_var'], u_var=1.0, g_var=1,
							optimiser_in=hyp['optimiser_in'], 
							learning_rate=hyp['learning_rate'], 
							hidden_size=config.units, 
							n_epochs=hyp['n_epochs'], 
							cycle_print=hyp['cycle_print'], 
							n_ensembles=config.n_models,
							total_trained=0,
							batch_size=hyp['batch_size'],
							decay_rate=hyp['decay_rate'],
							model_name=model_name+'_featureset_' + str(i)
							)
		
		y_priors, y_prior_mu, y_prior_std = ens.train(np.asarray(x_train[i]), np.asarray(y_train), np.asarray(x_val[i]), np.asarray(y_val), is_print=is_print)

		y_preds, _mu, _std = ens.predict(np.asarray(x_val[i]))
		
		all_features_ensemble_preds.append(y_preds)
	
		y_pred_mu = np.atleast_2d(np.mean(y_preds,axis=0)).T
		y_pred_std = np.atleast_2d(np.std(y_preds,axis=0, ddof=1)).T
		y_pred_std = np.sqrt(np.square(y_pred_std) + hyp['data_noise'])

		# anc_utils.metrics_calc(y_val, y_pred_mu, y_pred_std, scale_c, hyp['b_0_var'], hyp['w_0_var'], hyp['data_noise'], ens, is_print=True)

		cluster_rmse = np.sqrt(np.mean(np.square(scale_c*(y_val - y_pred_mu))))
		ensemble_clusterwise_val_rmse.append(cluster_rmse)
		print('Cluster {} RMSE: {:.3f}'.format(i, cluster_rmse))

		ensemble_sigmas.append(y_pred_std)

		feature_nll = anc_utils.gauss_neg_log_like(y_val, y_pred_mu, y_pred_std, scale_c=scale_c)
		featurewise_nll.append(feature_nll)
		print('Cluster {} NLL: {:.3f}'.format(i, feature_nll))
		print("-"*20)

		
	all_features_ensemble_preds = np.asarray(all_features_ensemble_preds)

	all_features_ensemble_preds_flip = np.swapaxes(all_features_ensemble_preds, 0, 1)
		
	mus=[]		
	for model_id in range(config.n_models):
		mu = all_features_ensemble_preds_flip[model_id]
		mu = np.sum(mu, axis=0) / config.n_feature_sets
		mus.append(mu)

	ensemble_mus = np.mean(mus, axis=0).reshape(-1,1)

	ensemble_val_rmse = np.sqrt(np.mean(np.square(scale_c*(y_val - ensemble_mus))))

	ensemble_val_nll = []
	for i in range(n_feature_sets):
		ensemble_val_nll.append(anc_utils.gauss_neg_log_like(y_val, ensemble_mus, ensemble_sigmas[i], scale_c=scale_c))

	print("Ensemble val rmse {}".format(ensemble_val_rmse))
	print("Ensemble val nlls {}".format(ensemble_val_nll)) # uses ensemble_mu
	print("Ensemble clusterwise val rmse {}".format(ensemble_clusterwise_val_rmse))

	print("Featurewise nll : {}".format(featurewise_nll)) # uses mu for the particular feature in the ensemble
	return ensemble_val_rmse, ensemble_val_nll, ensemble_clusterwise_val_rmse
