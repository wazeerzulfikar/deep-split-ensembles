import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import *

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import random
from scipy.interpolate import  make_interp_spline, BSpline

import os
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns


import models
import combined_uncertainty

tfd = tfp.distributions

def plot_toy_regression(config):
	config.units = 100
	fold = 0
	config.n_models = 5
	config.epochs = 6000
	config.lr = 0.1
	config.verbose = 1
	config.batch_size = 8
	toy='2d'
	graph_limits = [-5, 5]
	x_limits = [-4, 4]

	n_datapoints = 40
	x_1d = np.linspace(graph_limits[0], graph_limits[1], num=n_datapoints)

	x1 = np.random.uniform(x_limits[0], x_limits[1], size=(n_datapoints,1))
	e1 = np.random.normal(loc=0, scale=3, size=(n_datapoints,1))

	if toy == '3d':

		x2 = np.random.uniform(x_limits[0], x_limits[1], size=(n_datapoints,1))
		e2 = np.random.normal(loc=1, scale=2, size=(n_datapoints,1))

		x = [x1, x2]

		x1x1, x2x2 = np.meshgrid(range(graph_limits[0],graph_limits[1]+1), range(graph_limits[0],graph_limits[1]+1))
		y_2d = np.power(x1x1, 4) * np.power(x2x2, 4)
		y = (np.power(x1, 4) + e1) * (np.power(x2, 4) + e2)

		# y_1d_1, x_1d_1 = [] 
		# y_1d = np.power(x_1d, 3) ** 2

		y_2d = np.squeeze(y_2d)

	elif toy == '2d':
		x2 = np.random.uniform(x_limits[0], x_limits[1], size=(n_datapoints,1))
		e2 = np.random.normal(loc=1, scale=2, size=(n_datapoints,1))

		x = [x1, x2]
		y = np.power(x1, 3) + e1
		y = np.power(x1, 3) + e1
		y_1d = (np.power(x1, 3) + e1) * (np.power(x2, 3) + e2)
		y_1d = np.squeeze(y_1d)	

	y = np.squeeze(y)

	config.n_feature_sets = len(x)
	config.feature_split_lengths = [i.shape[1] for i in x]


	# model, _ = combined_uncertainty.train_a_model(fold, 0, x, y, x, y, config)
	combined_uncertainty.train_deep_ensemble(x, y, x, y, fold, config, train=True)

	mus, featurewise_sigmas = [], [[] for i in range(config.n_feature_sets)]
	for model_id in range(config.n_models):
		print(model_id)
		model, _ = models.build_model(config)
		model.load_weights(os.path.join(config.model_dir, 'fold_{}_nll_{}.h5'.format(fold, model_id)))

		pred = model([x_1d, x_1d])
		mus.append(pred.mean().numpy()[:,0])
		for i in range(config.n_feature_sets):
			featurewise_sigmas[i].append(pred.stddev().numpy()[:, i:i+1])

	ensemble_mus = np.mean(mus, axis=0).reshape(-1,1)
	ensemble_sigmas = []

	for i in range(config.n_feature_sets):
		ensemble_sigma = np.sqrt(np.mean(np.square(featurewise_sigmas[i]) + np.square(ensemble_mus), axis=0).reshape(-1,1) - np.square(ensemble_mus))
		ensemble_sigmas.append(ensemble_sigma)

	ensemble_mus = np.squeeze(ensemble_mus, axis=-1)
	ensemble_sigmas = np.squeeze(ensemble_sigmas, axis=-1)
	if toy == '3d':
		# 3D
		print(ensemble_sigmas)

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_wireframe(x1x1, x2x2, y_2d, rstride=1, cstride=1, alpha=0.3)
		# ax.plot_surface(x1x1, x2x2, y_, rstride=1, cstride=1, alpha=0.5)
		ax.scatter(x1, x2, y, s=[8]*len(x1), c='r', zorder=2, zdir='z')

		ax.scatter(x1, graph_limits[1], y, s=[8]*len(x1), c='g', zorder=2, zdir='z')
		ax.scatter(graph_limits[0], x2, y, s=[8]*len(x1), c='g', zorder=2, zdir='z')

		# ax.plot(x_1d, y_1d, zs=-6, zdir='x', color='blue')
		# ax.plot(x_1d, y_1d, zs=6, zdir='y', color='blue')

		ax.contourf(x1x1, x2x2, y_2d, zdir='x', offset=graph_limits[0], alpha=0.3, levels=0, colors='C0')
		ax.contourf(x1x1, x2x2, y_2d, zdir='y', offset=graph_limits[1], alpha=0.3, levels=0, colors='C0')

		# ax.contourf(np.zeros_like(x1x1), x2x2, y_2d, zdir='x', offset=-6, alpha=0.3, levels=0, colors='C0')
		# ax.contourf(x1x1, np.zeros_like(x2x2), y_2d, zdir='y', offset=6, alpha=0.3, levels=0, colors='C0')

		ax.add_collection3d(plt.fill_between(x_1d, ensemble_mus-3*ensemble_sigmas[1],
		 ensemble_mus+3*ensemble_sigmas[1], color='grey', alpha=0.3), zs=graph_limits[0], zdir='x')
		ax.add_collection3d(plt.fill_between(x_1d, ensemble_mus-3*ensemble_sigmas[0], 
			ensemble_mus+3*ensemble_sigmas[0], color='grey', alpha=0.3), zs=graph_limits[1], zdir='y')

		# ax.set_xlabel('x_1')
		# ax.set_ylabel('x_2')
		# ax.set_zlabel('y')

		# ax.set_title('Y = (X1^3)*(X2^3)')
		ax.set_xlim(graph_limits[0], graph_limits[1])
		# ax.set_ylabel('Y')
		ax.set_ylim(graph_limits[0], graph_limits[1])
		# ax.set_zlabel('Z')
		# ax.set_zlim(-200, 600)
		ax.set_zlim(-50000, 400000)

	if toy == '2d':
		print(ensemble_sigmas)

	# 2D
		plt.plot(x_1d, y_1d)
		plt.scatter(x1, y, s=[6]*len(x1), c='r', zorder=1)
		plt.fill_between(x_1d, np.squeeze(ensemble_mus-3*ensemble_sigmas[0]), 
			np.squeeze(ensemble_mus+3*ensemble_sigmas[0]), color='grey', alpha=0.5)
	plt.show()

def plot_calibration(X, y, config):
	ensemble_mus, ensemble_sigmas, true_values, _ = get_ensemble_predictions(X, y, ood=False, config=config)
	fig = plt.figure()
	# fig.set_size_inches(30.,18.)

	for i in range(config.n_feature_sets):
		print('feature set {}'.format(i))
		defered_rmse_list, non_defered_rmse_list = defer_analysis(true_values, ensemble_mus, ensemble_sigmas[...,i])

		total_samples = ensemble_mus.shape[0]
		
		drop_n = int(0.1*ensemble_mus.shape[0])

		use_samples = total_samples - drop_n 
		non_defered_rmse_list = non_defered_rmse_list[:-drop_n]
		plt.plot(range(use_samples+1), non_defered_rmse_list, label='Cluster '+str(i+1))
		plt.legend(loc='lower left')
		plt.xlabel('No. of Datapoints Deferred')
		# plt.ylabel('Root Mean Squared Error')
		plt.xticks(range(0, use_samples+1, (use_samples)//10))
		plt.title(config.dataset.capitalize().replace('_',' '))

		if config.dataset == 'boston':
			plt.title('Boston housing')
		if config.dataset == 'energy_efficiency':
			plt.title('Energy')


	plt.tight_layout()
	plt.savefig(config.plot_name, dpi=1200)
	plt.show()


def plot_ood(X, y, config):
	ensemble_mus, ensemble_sigmas, true_values, ensemble_entropies = get_ensemble_predictions(X, y, ood=0, config=config)
	
	plot_ood_helper(ensemble_entropies, os.path.join(config.plot_name,'in.png'), config)

	ensemble_mus, ensemble_sigmas, true_values, ensemble_entropies = get_ensemble_predictions(X, y, ood=1, config=config)
	
	plot_ood_helper(ensemble_entropies, os.path.join(config.plot_name,'out_1.png'), config)

	ensemble_mus, ensemble_sigmas, true_values, ensemble_entropies = get_ensemble_predictions(X, y, ood=2, config=config)
	
	plot_ood_helper(ensemble_entropies, os.path.join(config.plot_name,'out_2.png'), config)

	ensemble_mus, ensemble_sigmas, true_values, ensemble_entropies = get_ensemble_predictions(X, y, ood=3, config=config)
	
	plot_ood_helper(ensemble_entropies, os.path.join(config.plot_name,'out_3.png'), config)

def plot_ood_helper(ensemble_entropies, plot_name, config):
	for i in range(config.n_feature_sets):
		print('feature set {}'.format(i))

		sns.distplot(ensemble_entropies[...,i], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'Cluster '+str(i+1))
		# plt.hist(ensemble_entropies[i], label='Cluster '+str(i), bins=150)

		plt.legend(loc='upper right')

	plt.xlim(0,6)

	if config.dataset == 'boston':
		plt.ylim(0,3)

	if config.dataset == 'energy_efficiency':
		plt.ylim(0,9)

	if config.dataset == 'wine':
		plt.ylim(0,13)

	if config.dataset == 'cement':
		plt.ylim(0,7)

	plt.xlabel('Entropy Values')
	plt.ylabel('Density')
	plt.savefig(plot_name)
	# plt.show()
	plt.clf()


def standard_scale(x_train, x_test):
	scalar = StandardScaler()
	scalar.fit(x_train)
	x_train = scalar.transform(x_train)
	x_test = scalar.transform(x_test)
	return x_train, x_test

def get_ensemble_predictions(X, y, ood=False, config=None):
	kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
	fold = 1
	all_mus, all_sigmas, true_values, all_entropies = [], [], [], []
	n_feature_sets = len(X)
	for train_index, test_index in kf.split(y):
		# if fold == fold_to_use:
		print('Fold ', fold)
		y_train, y_val = y[train_index], y[test_index]
		x_train = [i[train_index] for i in X]
		x_val = [i[test_index] for i in X]


		for i in range(n_feature_sets):
			x_train[i], x_val[i] = standard_scale(x_train[i], x_val[i])

		if ood == 1:
			x_val[0][:,0] = np.random.normal(loc=6, scale=2, size=x_val[0][:,0].shape)
		if ood == 2:
			x_val[1][:,0] = np.random.normal(loc=6, scale=2, size=x_val[1][:,0].shape)
		if ood == 3:
			x_val[0][:,0] = np.random.normal(loc=6, scale=2, size=x_val[0][:,0].shape)
			x_val[1][:,0] = np.random.normal(loc=12, scale=1, size=x_val[1][:,0].shape)

		mus = []
		featurewise_entropies = [[] for i in range(n_feature_sets)]
		featurewise_sigmas = [[] for i in range(n_feature_sets)]
		for model_id in range(config.n_models):
			model, _ = models.build_model(config)
			model.load_weights(os.path.join(config.model_dir, 'fold_{}_nll_{}.h5'.format(fold, model_id)))

			y_val = y_val.reshape(-1,1)
			preds = model(x_val)

			if config.build_model == 'combined_multivariate':
				mus.append(preds.mean().numpy()[:,0])
			elif config.build_model == 'combined_pog':
				mus.append(preds[0].mean().numpy())

			for i in range(n_feature_sets):
				if config.build_model == 'combined_multivariate':
					featurewise_sigmas[i].append(preds.stddev().numpy()[:,i:i+1])
					featurewise_entropies[i].append(preds.entropy().numpy())

				elif config.build_model == 'combined_pog':
					featurewise_sigmas[i].append(preds[i].stddev().numpy())
					featurewise_entropies[i].append(preds[i].entropy().numpy())


		ensemble_mus = np.mean(mus, axis=0).reshape(-1,1)
		ensemble_entropies = np.mean(featurewise_entropies, axis=1).reshape(n_feature_sets, -1)
		ensemble_sigmas = []
		for i in range(n_feature_sets):
			ensemble_sigma = np.sqrt(np.mean(np.square(featurewise_sigmas[i]) + np.square(ensemble_mus), axis=0).reshape(-1,1) - np.square(ensemble_mus))
			ensemble_sigmas.append(ensemble_sigma)

		for i in range(y_val.shape[0]):
			all_mus.append(ensemble_mus[i])
			all_sigmas.append([ensemble_sigmas[j][i] for j in range(n_feature_sets)])
			all_entropies.append([ensemble_entropies[j][i] for j in range(n_feature_sets)])
			true_values.append(y_val[i])
		fold+=1
		val_rmse = mean_squared_error(y_val, ensemble_mus, squared=False)
		print('Val RMSE: {:.3f}'.format(val_rmse))

		if config.dataset in ['msd', 'alzheimers']:
			break

	all_mus = np.reshape(all_mus, (-1,1))
	all_sigmas = np.reshape(all_sigmas, (-1, n_feature_sets))
	true_values = np.reshape(true_values, (-1, 1))
	all_entropies = np.reshape(all_entropies, (-1, n_feature_sets))

	return all_mus, all_sigmas, true_values, all_entropies

def defer_analysis(true_values, predictions, defer_based_on):

	defered_rmse_list, non_defered_rmse_list = [], []
	defer_based_arg_sorted = np.argsort(defer_based_on)
	true_values_sorted = true_values[defer_based_arg_sorted]
	predictions_sorted = predictions[defer_based_arg_sorted]
	for i in range(predictions.shape[0]+1):
		# if i==predictions.shape[0]:
		# 	defered_rmse = mean_squared_error(true_values, predictions, squared=False)
		# elif i==0:
		# 	defered_rmse = 0
		# else:
		# 	defered_rmse = mean_squared_error(
		# 		true_values[np.argsort(defer_based_on)][-i:], 
		# 		predictions[np.argsort(defer_based_on)][-i:], squared=False)
		# defered_rmse_list.append(defered_rmse)

		if i==0:
			non_defered_rmse = mean_squared_error(true_values, predictions, squared=False)
		elif i==predictions.shape[0]:
			non_defered_rmse = 0
		else:
			non_defered_rmse = mean_squared_error(
				true_values_sorted[:-i], 
				predictions_sorted[:-i], squared=False)

		non_defered_rmse_list.append(non_defered_rmse)
		# print('\n{} datapoints deferred'.format(i))

		# print('Defered RMSE : {:.3f}'.format(defered_rmse))
		# print('Not Defered RMSE : {:.3f}'.format(non_defered_rmse))
	return defered_rmse_list, non_defered_rmse_list