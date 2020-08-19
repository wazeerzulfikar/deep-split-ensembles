import os
import numpy as np
np.random.seed(0)

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns

import models
import trainer
import dataset
import utils
from extras import evaluator
from alzheimers import alz_utils as alzheimers_utils

tfd = tfp.distributions

def plot_defer_simulation(X, y, config):

	ensemble_mus, ensemble_sigmas, true_values, _ = get_ensemble_predictions(X, y, ood=False, config=config)

	for i in range(config.n_feature_sets):
		print('feature set {}'.format(i))
		defered_rmse_list, non_defered_rmse_list = utils.defer_analysis(true_values, ensemble_mus, ensemble_sigmas[...,i])

		total_samples = ensemble_mus.shape[0]
		
		drop_n = int(0.1*ensemble_mus.shape[0])

		use_samples = total_samples - drop_n 
		non_defered_rmse_list = non_defered_rmse_list[:-drop_n]
		plt.plot(range(use_samples+1), non_defered_rmse_list, label='Cluster '+str(i+1), linewidth=3)
		plt.legend(loc='lower left', fontsize=14)
		plt.xlabel('No. of Datapoints Deferred', fontsize=22)
		plt.xticks(range(0, use_samples+1, (use_samples)//10))
		plt.title(config.dataset.capitalize().replace('_',' ') + ' (hier. clust.)', fontsize=24)

		if config.dataset == 'boston':
			plt.title('Boston (hier. clust.)', fontsize=22)
		if config.dataset == 'energy_efficiency':
			plt.title('Energy'+' (hier. clust.)', fontsize=22)
		if config.dataset == 'cement':
			plt.title('Concrete'+' (hier. clust.)', fontsize=22)
		if config.dataset == 'power_plant':
			plt.title('Power'+' (hier. clust.)', fontsize=22)

	plt.ylabel('Non-Deferred RMSE', fontsize=22)

	plt.tight_layout(pad=0.0)
	plt.savefig(config.plot_name, dpi=300)
	plt.clf()
	plt.close()


def plot_kl(X, y, config):

	# ensemble_mus, ensemble_sigmas, true_values, ensemble_entropies = get_ensemble_predictions(X, y, ood=0, config=config)

	fig, ax = plt.subplots()

	for cluster_id in range(config.n_feature_sets):

		kl_1= []
		entropy_mode_1 = []

		loc_values = [2, 4, 6, 8, 10, 12]
		scale_values = [1.5, 1.5, 1, 1, 0.5, 0.5]

		ood_params = [r"$\mathcal{N}(2,1.5^2)$", r"$\mathcal{N}(4,1.5^2)$", r"$\mathcal{N}(6,1^2)$",
		r"$\mathcal{N}(8,1^2)$", r"$\mathcal{N}(10,0.5^2)$", r"$\mathcal{N}(12,0.5^2)$"]

		for mu, sigma in zip(loc_values, scale_values):
			ensemble_mus, ensemble_sigmas, true_values, ensemble_entropies = get_ensemble_predictions(X, y, ood=100, 
				config=config, ood_loc=mu, ood_scale=sigma, ood_cluster_id=cluster_id)

			hist, bins = np.histogram(ensemble_entropies[..., cluster_id], bins = 30)
			kl_1.append(tfd.Normal(loc=0,scale=1).kl_divergence(tfd.Normal(loc=mu,scale=sigma)))
			entropy_mode_1.append(np.mean(bins[np.argmax(hist):np.argmax(hist)+2]))

			x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

		# np.save(config.plot_name.replace('.png', '_{}'.format(cluster_id)), np.stack((kl_1, entropy_mode_1), axis=-1))

		# plot_folder = config.plot_name.split('/')[0]
		# data = np.load(os.path.join(plot_folder, 'kl_plots_values', '{}_{}.npy'.format(config.dataset, cluster_id)))
		# kl_1, entropy_mode_1 = data[...,0], data[...,1]

		kl_sorted_ind_1 = np.argsort(kl_1)
		kl_1 = np.array(kl_1)[kl_sorted_ind_1]
		entropy_mode_1 = np.array(entropy_mode_1)[kl_sorted_ind_1]
		ood_params = np.array(ood_params)[kl_sorted_ind_1]

		plt.scatter(kl_1, entropy_mode_1, s=96)
		if cluster_id <3:
			plt.plot(kl_1, entropy_mode_1, label='Cluster '+str(cluster_id+1), linewidth=3)
		else:
			plt.plot(kl_1, entropy_mode_1, label='Cluster '+str(cluster_id), linewidth=3)

		plt.title(config.dataset.capitalize().replace('_',' ')+' (hier. clust.)', fontsize=26)
		
		plt.ylabel('Mode of KDE of Entropy', fontsize=26)

		if config.dataset == 'boston':
			plt.title('Boston'+' (hier. clust.)', fontsize=26)

		if config.dataset == 'energy_efficiency':
			plt.title('Energy'+' (hier. clust.)', fontsize=26)
		if config.dataset == 'cement':
			plt.title('Concrete'+' (hier. clust.)', fontsize=26)

		if config.dataset == 'power_plant':
			plt.title('Power'+' (human experts)', fontsize=26)

		plt.xlabel(r'$D_{KL}$ (In Dist. || OOD)', fontsize=26)

	ax.tick_params(axis="x", labelsize=24)
	ax.tick_params(axis="y", labelsize=24)
	plt.legend(fontsize=16, loc='lower right')
	plt.tight_layout(pad=0)
	plt.savefig(config.plot_name, dpi=300)
	plt.clf()
	plt.close()


def plot_ood(X, y, config):

	if 'alzheimers' in config.dataset:
		plot_alzheimers_ood(X, y, config)
		return

	# ensemble_entropies = np.concatenate(np.load('plots/clusterwise_ood/deep_ensemble/{}_val_entropy.npy'.format(config.dataset), 
	# 	allow_pickle=True))
	config.plot_name = config.plot_name.replace('.png', '')
	os.makedirs(config.plot_name, exist_ok=True)

	# ensemble_entropies = np.squeeze(ensemble_entropies)
	# plot_ood_helper(ensemble_entropies, os.path.join(config.plot_name,'de.png'), config, deep_ensemble=True)

	ensemble_mus, ensemble_sigmas, true_values, ensemble_entropies = get_ensemble_predictions(X, y, config=config, ood=0)
	
	plot_ood_helper(ensemble_entropies, os.path.join(config.plot_name,'in.png'), config)

	ensemble_mus, ensemble_sigmas, true_values, ensemble_entropies = get_ensemble_predictions(X, y, config=config, ood=1)
	
	plot_ood_helper(ensemble_entropies, os.path.join(config.plot_name,'out_1.png'), config, ood=1)

	ensemble_mus, ensemble_sigmas, true_values, ensemble_entropies = get_ensemble_predictions(X, y, config=config, ood=2)
	
	plot_ood_helper(ensemble_entropies, os.path.join(config.plot_name,'out_2.png'), config, ood=2)

	ensemble_mus, ensemble_sigmas, true_values, ensemble_entropies = get_ensemble_predictions(X, y, config=config, ood=3)
	
	plot_ood_helper(ensemble_entropies, os.path.join(config.plot_name,'out_3.png'), config, ood=3)

def plot_ood_helper(ensemble_entropies, plot_name, config, deep_ensemble=False, ood=False):

	if ood == False:
		plt.plot([], [], ' ', label="In Dist.", color='white')

	if deep_ensemble:
		b = sns.distplot(ensemble_entropies, hist = False, kde = True,
                 kde_kws = {'linewidth': 3}, label='Unified', color='black')
		b.set_ylabel('Density', fontsize=26)
	else:
		ood_legend_params = [r'In Dist.', r'OOD $\mathcal{N}(6,2^2)$', r'OOD $\mathcal{N}(12,1^2)$']
		for i in range(config.n_feature_sets):
			print('feature set {}'.format(i))
			if ood == False:
				b = sns.distplot(ensemble_entropies[...,i], hist = False, kde = True,
		                 kde_kws = {'linewidth': 3},
		                 label = 'Cluster '+str(i+1))
			else:
				if (ood == 1 and i == 0) or (ood == 3 and i == 0) or (ood == 2 and i == 1):

					b = sns.distplot(ensemble_entropies[...,i], hist = False, kde = True,
			                 kde_kws = {'linewidth': 3},
			                 label = ood_legend_params[1])
					b.lines[i].set_linestyle("--")

				elif ood == 3 and i == 1:
					b = sns.distplot(ensemble_entropies[...,i], hist = False, kde = True,
			                 kde_kws = {'linewidth': 3},
			                 label = ood_legend_params[2])
					b.lines[i].set_linestyle(":")

				else:
					b = sns.distplot(ensemble_entropies[...,i], hist = False, kde = True,
			                 kde_kws = {'linewidth': 3},
			                 label = ood_legend_params[0])

	b.legend(loc='upper right', fontsize=16)

	plt.xlim(0,6)

	if config.dataset == 'boston':
		plt.ylim(0,4)

	if config.dataset == 'energy_efficiency':
		plt.ylim(0,9)

	if config.dataset == 'wine':
		# plt.ylim(0,8)
		plt.ylim(0,12)

	if config.dataset == 'cement':
		plt.ylim(0,7)

	if config.dataset == 'power_plant':
		# plt.ylim(0,6)
		plt.ylim(0,10)

	if config.dataset == 'kin8nm':
		plt.ylim(0,5)
		plt.xlim(-4,6)


	if config.dataset == 'yacht':
		plt.ylim(0,6)

	if config.dataset == 'protein':
		plt.ylim(0,7)

	b.tick_params(labelsize=24)
	b.set_xlabel('Entropy (Nats)', fontsize=26)

	plt.tight_layout(pad=0)
	plt.savefig(plot_name, dpi=300)
	plt.clf()
	plt.close()

def plot_alzheimers_ood(X, y, config):

	# ensemble_mus, ensemble_sigmas, true_values, ensemble_entropies = get_ensemble_predictions(X, y, ood=0, 
	# 	config=config, alzheimers_test_data='alzheimers_test')
	# plot_alzheimers_ood_helper(ensemble_entropies, os.path.join(config.plot_name,'in.png'), config, ood=0)

	ensemble_mus, ensemble_sigmas, true_values, ensemble_entropies = get_ensemble_predictions(X, y, ood=0, 
		config=config, alzheimers_test_data='alzheimers_test_female')
	plot_alzheimers_ood_helper(ensemble_entropies, os.path.join(config.plot_name,'in_F.png'), config, ood=0)

	ensemble_mus, ensemble_sigmas, true_values, ensemble_entropies = get_ensemble_predictions(X, y, ood=0, 
		config=config, alzheimers_test_data='alzheimers_test_male')
	plot_alzheimers_ood_helper(ensemble_entropies, os.path.join(config.plot_name,'out_M.png'), config, ood=1)


def plot_alzheimers_ood_helper(ensemble_entropies, plot_name, config ,ood=0):

	feature_sets = ['Interventions', 'Disfluency', 'Acoustic']

	if ood == False:
		plt.ylabel('Density', fontsize=26)

	for i in range(config.n_feature_sets):
		print('feature set {}'.format(i))
		b = sns.distplot(ensemble_entropies[...,i], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = feature_sets[i])
	
	plt.title('Alzheimers (female)', fontsize=26)

	b.tick_params(labelsize=24)
	b.set_xlabel('Entropy (Nats)', fontsize=26)
	plt.xlim(0,12)
	plt.xticks(range(0,12,2), fontsize=24)
	plt.yticks(range(0,6,1), fontsize=24)
	plt.legend(fontsize=20, loc='upper right')

	plt.tight_layout(pad=0)
	plt.savefig(plot_name, dpi=300)
	plt.clf()
	plt.close()


def plot_toy_regression(config):
	config.units = 100
	fold = 0
	config.n_models = 5
	config.epochs = 200
	config.lr = 0.1
	config.batch_size = 8
	toy='3d'
	graph_limits = [-5, 5]
	x_limits = [-4, 4]

	n_datapoints = 40
	x_1d = np.linspace(graph_limits[0], graph_limits[1], num=n_datapoints)

	x1 = np.random.uniform(x_limits[0], x_limits[1], size=(n_datapoints,1))
	e1 = np.random.normal(loc=0, scale=3, size=(n_datapoints,1))

	power = config.power

	if toy == '3d':

		x2 = np.random.uniform(x_limits[0], x_limits[1], size=(n_datapoints,1))
		e2 = np.random.normal(loc=1, scale=2, size=(n_datapoints,1))

		x = [x1, x2]

		x1x1, x2x2 = np.meshgrid(range(graph_limits[0],graph_limits[1]+1), range(graph_limits[0],graph_limits[1]+1))
		y_2d = np.power(x1x1, power) * np.power(x2x2, power)
		y = (np.power(x1, power) + e1) * (np.power(x2, power) + e2)

		y_2d = np.squeeze(y_2d)

	elif toy == '2d':
		x2 = np.random.uniform(x_limits[0], x_limits[1], size=(n_datapoints,1))
		e2 = np.random.normal(loc=1, scale=2, size=(n_datapoints,1))

		x = [x1, x2]
		y = (np.power(x1, 4) + e1) * (np.power(x2, 4) + e2)
		x_1d = np.expand_dims(x_1d, axis=-1)
		y_1d = (np.power(x_1d, 3) + e1) 
		y_1d = np.squeeze(y_1d)	

	y = np.squeeze(y)

	if config.build_model == 'gaussian':
		x = np.squeeze(x)
		x = np.transpose(x)
		y = np.expand_dims(y, axis=-1)
		print(x.shape)
		print(y.shape)
		mus, sigmas = [], []
		for model_id in range(config.n_models):
			model, _ = evaluator.train_a_fold(0, model_id, config, x, y, x, y)
			model, _ = models.build_model(config)
			model.build((None, 2))
			model.load_weights(os.path.join(config.model_dir,'fold_{}_nll_{}.h5'.format(0, model_id)))
			preds = model(np.stack([x_1d, x_1d], axis=-1))
			mus.append(preds.mean().numpy())
			sigmas.append(preds.stddev().numpy())

		ensemble_mus = np.mean(mus, axis=0).reshape(-1,1)

		ensemble_sigmas = np.sqrt(np.mean(np.square(sigmas) + np.square(ensemble_mus), axis=0).reshape(-1,1) - np.square(ensemble_mus))

		ensemble_mus = np.squeeze(ensemble_mus, axis=-1)
		ensemble_sigmas = np.squeeze(ensemble_sigmas, axis=-1)

		ensemble_sigmas = np.stack((ensemble_sigmas, ensemble_sigmas), axis=0)

	else:

		config.n_feature_sets = len(x)
		config.feature_split_lengths = [i.shape[1] for i in x]


		model, _ = trainer.train_a_model(fold, 0, x, y, x, y, config)
		trainer.train_deep_ensemble(x, y, x, y, fold, config, train=True)

		mus, featurewise_sigmas = [], [[] for i in range(config.n_feature_sets)]
		for model_id in range(config.n_models):
			print(model_id)
			model, _ = models.build_model(config)
			model.load_weights(os.path.join(config.model_dir, 'fold_{}_nll_{}.h5'.format(fold, model_id)))

			pred = model([x_1d, x_1d])
			mus.append(pred[0].mean().numpy())
			for i in range(config.n_feature_sets):
				featurewise_sigmas[i].append(pred[i].stddev().numpy())

		ensemble_mus = np.mean(mus, axis=0).reshape(-1,1)
		ensemble_sigmas = []

		for i in range(config.n_feature_sets):
			ensemble_sigma = np.sqrt(np.mean(np.square(featurewise_sigmas[i]) + np.square(ensemble_mus), axis=0).reshape(-1,1) - np.square(ensemble_mus))
			ensemble_sigmas.append(ensemble_sigma)

		ensemble_mus = np.squeeze(ensemble_mus, axis=-1)
		ensemble_sigmas = np.squeeze(ensemble_sigmas, axis=-1)
	if toy == '3d':
		# 3D

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.set_zticks([], [])

		ax.plot_wireframe(x1x1, x2x2, y_2d, rstride=1, cstride=1, alpha=0.3)
		ax.scatter(x1, x2, y, s=16, c='red', zorder=2, zdir='z', depthshade=False)

		ax.scatter(x1, graph_limits[1], y, s=8, c='black', zorder=2, zdir='z', depthshade=False)
		ax.scatter(graph_limits[0], x2, y, s=8, c='black', zorder=2, zdir='z', depthshade=False)

		# ax.plot(x_1d, y_1d, zs=-6, zdir='x', color='blue')
		# ax.plot(x_1d, y_1d, zs=6, zdir='y', color='blue')

		# ax.contourf(x1x1, x2x2, y_2d, zdir='x', offset=graph_limits[0], alpha=0.3, levels=0, colors='C0')
		# ax.contourf(x1x1, x2x2, y_2d, zdir='y', offset=graph_limits[1], alpha=0.3, levels=0, colors='C0')

		# ax.contourf(np.zeros_like(x1x1), x2x2, y_2d, zdir='x', offset=-6, alpha=0.3, levels=0, colors='C0')
		# ax.contourf(x1x1, np.zeros_like(x2x2), y_2d, zdir='y', offset=6, alpha=0.3, levels=0, colors='C0')

		ax.add_collection3d(plt.fill_between(x_1d, ensemble_mus-3*ensemble_sigmas[1],
		 ensemble_mus+3*ensemble_sigmas[1], color='grey', alpha=0.3), zs=graph_limits[0], zdir='x')
		ax.add_collection3d(plt.fill_between(x_1d, ensemble_mus-3*ensemble_sigmas[0], 
			ensemble_mus+3*ensemble_sigmas[0], color='grey', alpha=0.3), zs=graph_limits[1], zdir='y')

		ax.set_xlabel(r'$x_1$', fontsize=26)
		ax.set_ylabel(r'$x_2$', fontsize=26)
		ax.set_zlabel(r'$y$', fontsize=26)
		if power == 3:
			ax.set_title(r'$y=(x_1^3+\epsilon_1)(x_2^3+\epsilon_2)$', fontsize=26)
		elif power == 4:
			ax.set_title(r'$y=(x_1^4+\epsilon_1)(x_2^4+\epsilon_2)$', fontsize=26)

		ax.set_xlim(graph_limits[0], graph_limits[1])
		ax.set_ylim(graph_limits[0], graph_limits[1])
		if power == 4:
			ax.set_zlim(-50000, 400000)

		ax.grid(False)

	if toy == '2d':
		plt.plot(x_1d, y_1d)
		plt.scatter(x1, y, s=[6]*len(x1), c='r', zorder=1)
		plt.fill_between(x_1d, np.squeeze(ensemble_mus-3*ensemble_sigmas[0]), 
			np.squeeze(ensemble_mus+3*ensemble_sigmas[0]), color='grey', alpha=0.5)


	plt.tight_layout(pad=0)
	plt.savefig(config.plot_name, dpi=300)


def show_model_summary(X, y, config):

	n_feature_sets = len(X)
	model, loss= models.build_model(config)
	model.compile(loss=loss, optimizer='adam')
	model.build((None,np.concatenate(X, axis=-1).shape[1]))
	print(model.summary())

def empirical_rule_test(X, y, config):
	mus, sigmas, true_values, _ = get_ensemble_predictions(X, y, config=config)

	thresholds = [0.12566, 0.25335, 0.38532, 0.52440, 0.67339, 0.84162, 1.03643, 1.28155, 1.64485]

	for cluster_id in range(len(sigmas[0])):
		print('Cluster {}'.format(cluster_id+1))
		count_1 = 0
		count_2 = 0
		count_3 = 0
		for i in range(len(mus)):
			if np.abs(mus[i] - true_values[i])< sigmas[i,cluster_id]:
				count_1+=1 
			if np.abs(mus[i] - true_values[i])< 2*sigmas[i,cluster_id]:
				count_2+=1 
			if np.abs(mus[i] - true_values[i])< 3*sigmas[i,cluster_id]:
				count_3+=1 

		print('1 std - {} out of {} - {:.3}%'.format(count_1, len(mus), (count_1*100/len(mus))))
		print('2 std - {} out of {} - {:.3}%'.format(count_2, len(mus), (count_2*100/len(mus))))
		print('3 std - {} out of {} - {:.3}%'.format(count_3, len(mus), (count_3*100/len(mus))))

	values = [[] for i in range(len(sigmas[0]))]
	threshold_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]

	fig, ax = plt.subplots()
	ideal = [i for i in range(10,100,10)]

	plt.plot(ideal, ideal, label='Ideal Calibration', linewidth=2, color='black', linestyle=':')


	for cluster_id in range(len(sigmas[0])):
		print('Cluster {}'.format(cluster_id+1))
		for t in thresholds:
			count = 0
			for i in range(len(mus)):
				if np.abs(mus[i] - true_values[i])< t* sigmas[i,cluster_id]:
					count+=1 
			values[cluster_id].append(count)

		values[cluster_id] = np.array(values[cluster_id])*100/len(mus)
		plt.scatter(threshold_values, values[cluster_id], s=96)
		plt.plot(threshold_values, values[cluster_id], label='Cluster '+str(cluster_id+1), linewidth=3)

		plt.title(config.dataset.capitalize().replace('_',' ')+' (hier. clust.)', fontsize=26)
		
		plt.ylabel('% of True Values inside Interval', fontsize=20)

		if config.dataset == 'boston':
			plt.title('Boston'+' (hier. clust.)', fontsize=26)

		if config.dataset == 'energy_efficiency':
			plt.title('Energy'+' (hier. clust.)', fontsize=26)
		if config.dataset == 'cement':
			plt.title('Concrete'+' (hier. clust.)', fontsize=26)

		if config.dataset == 'power_plant':
			plt.title('Power'+' (hier. clust.)', fontsize=26)

		plt.xlabel('% of Prediction Interval around Mean', fontsize=20)


	plt.xticks(range(10, 100, 10))
	plt.yticks(range(10, 100, 10))
	ax.tick_params(axis="x", labelsize=18)
	ax.tick_params(axis="y", labelsize=18)
	plt.legend(fontsize=16, loc='lower right')
	plt.tight_layout(pad=0)
	plt.savefig(config.plot_name, dpi=300)
	plt.clf()
	plt.close()

def get_ensemble_predictions(X, y, ood=False, config=None, ood_loc=0, ood_scale=1, ood_cluster_id=0,
 alzheimers_test_data=None):
	kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
	fold = 1
	all_mus, all_sigmas, true_values, all_entropies, all_rmses = [], [], [], [], []
	gaussian_split_mus = []
	n_feature_sets = len(X)
	for train_index, test_index in kf.split(y):
		y_train, y_val = y[train_index], y[test_index]
		x_train = [i[train_index] for i in X]
		x_val = [i[test_index] for i in X]

		if alzheimers_test_data is not None:
			config.dataset = alzheimers_test_data
			alzheimers_test_data = dataset._alzheimers_test(config)
			x_val = [np.array(alzheimers_test_data['{}'.format(i)]) for i in range(n_feature_sets)]
			y_val = np.array(alzheimers_test_data['y'])
			print('Alzheimers Testing..')
			[print('Shape of feature set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(x_val)]

		if 'alzheimers' in config.dataset:
			assert x_train[-1].shape[-1] == 6373, 'not compare'
			x_train[-1], x_val[-1] = alzheimers_utils.normalize_compare_features(x_train[-1], x_val[-1])

		else:
			for i in range(n_feature_sets):
				x_train[i], x_val[i] = utils.standard_scale(x_train[i], x_val[i])

		if ood == 1:
			x_val[0][:,0] = np.random.normal(loc=6, scale=2, size=x_val[0][:,0].shape)
		if ood == 2:
			x_val[1][:,0] = np.random.normal(loc=6, scale=2, size=x_val[1][:,0].shape)
		if ood == 3:
			x_val[0][:,0] = np.random.normal(loc=6, scale=2, size=x_val[0][:,0].shape)
			x_val[1][:,0] = np.random.normal(loc=12, scale=1, size=x_val[1][:,0].shape)
		if ood == 100:
			x_val[ood_cluster_id][:,0] = np.random.normal(loc=ood_loc, scale=ood_scale, size=x_val[ood_cluster_id][:,0].shape)

		mus = []
		featurewise_entropies = [[] for i in range(n_feature_sets)]
		featurewise_sigmas = [[] for i in range(n_feature_sets)]
		for model_id in range(config.n_models):
			if config.build_model == 'gaussian' and config.mod_split == 'none':
				model, _ = models.build_model(config)
				model.build((None,x_train[0].shape[1]))
				model.load_weights(os.path.join(config.model_dir, 'fold_{}_model_{}.h5'.format(fold, model_id+1)))
				preds = model(x_val[0])

			elif config.build_model == 'gaussian' and config.mod_split != 'none':
				gaussian_split_models = []
				for i in range(config.n_feature_sets):
					config.input_feature_length = config.feature_split_lengths[i]
					new_model_id = str(model_id)+'_'+str(i)
					model, _ = models.build_model(config)
					model.load_weights(os.path.join(config.model_dir, 'fold_{}_nll_{}.h5'.format(fold, new_model_id)))
					gaussian_split_models.append(model)

				gaussian_split_preds = []
				for i in range(config.n_feature_sets):
					gaussian_split_preds.append(gaussian_split_models[i](x_val[i]))

			else:

				model, _ = models.build_model(config)
				model.load_weights(os.path.join(config.model_dir, 'fold_{}_nll_{}.h5'.format(fold, model_id)))

				preds = model(x_val)

			y_val = y_val.reshape(-1,1)

			if config.build_model == 'gaussian' and config.mod_split != 'none':
				mu = [gaussian_split_preds[i].mean().numpy()[:,0] for i in range(config.n_feature_sets)]
				gaussian_split_mus.append(mu)
				mu = np.sum(mu, axis=0) / config.n_feature_sets
				mus.append(mu)

			elif config.build_model in ['combined_multivariate', 'gaussian']:
				mus.append(preds.mean().numpy()[:,0])

			elif config.build_model == 'combined_pog':
				mus.append(preds[0].mean().numpy())

			for i in range(n_feature_sets):
				if config.build_model == 'gaussian' and config.mod_split != 'none':
					featurewise_sigmas[i].append(gaussian_split_preds[i].stddev().numpy())
					featurewise_entropies[i].append(gaussian_split_preds[i].entropy().numpy())

				elif config.build_model in ['combined_multivariate', 'gaussian']:
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
		all_rmses.append(val_rmse)
		if config.verbose > 0:
			print('Fold ', fold-1)
			print('Val RMSE: {:.3f}'.format(val_rmse))

		if config.dataset == 'msd' or 'alzheimers' in config.dataset:
			break

		if config.dataset == 'naval' and fold == 6:
			break

	all_mus = np.reshape(all_mus, (-1,1))
	all_sigmas = np.reshape(all_sigmas, (-1, n_feature_sets))
	true_values = np.reshape(true_values, (-1, 1))
	all_entropies = np.reshape(all_entropies, (-1, n_feature_sets))

	print('Total val rmse', np.mean(all_rmses))

	return all_mus, all_sigmas, true_values, all_entropies