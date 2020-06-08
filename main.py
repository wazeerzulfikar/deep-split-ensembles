import numpy as np
np.random.seed(0)

import trainer
import dataset
import experiments
import utils

config = utils.EasyDict({
	'model_dir' : 'deepmind-1000/boston-1000',
	# 'model_dir' : 'deep_ensemble_models/boston',
	# 'model_dir' : 'alzheimers/alzheimers_models/alzheimers-5000-e-3-bs-16',
	# 'model_dir' : 'human/power_plant-1000',
	# 'model_dir': 'toys/toy2d',

	'regression_datasets_dir': 'datasets',
	'dataset': 'boston',

	# 'action': 'train',
	'action': 'evaluate',
	# 'action': 'plot_calibration',
	# 'action': 'plot_ood',
	# 'action': 'plot_kl',
	# 'action': 'show',
	# 'action': 'toy_regression',

	'build_model': 'combined_pog',
	# 'build_model': 'combined_multivariate',
	# 'build_model': 'gaussian',

	'mod_split': 'computation_split',
	# 'mod_split': 'human',
	# 'mod_split': 'none',

	'n_folds': 20,
	'n_models' :  5,

	'units_type': 'absolute',
	# 'units_type': 'prorated',

	'mixture_approximation': 'gaussian',
	# 'mixture_approximation': 'none',

	'verbose': 0,

	'lr': 0.01,
	'epochs': 1500,
	'batch_size': 8
	})

def main(config):
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

	if config.dataset in ['boston', 'cement', 'power_plant', 'wine', 'yacht', 'kin8nm', 'energy_efficiency', 'naval']:
		config.units = 50
	elif config.dataset in ['msd', 'protein', 'toy']:
		config.units = 100
	elif config.dataset in ['alzheimers', 'alzheimers_test']:
		config.units = 100
		config.feature_split_lengths[-1] = 21 # COMPARE features after PCA
		config.n_folds = 5

	if config.dataset == 'protein':
		config.n_folds = 5
	if config.dataset == 'msd':
		config.n_models = 2

	if config.action == 'train':
		print('Training..')
		trainer.train(X, y, config)

	elif config.action == 'evaluate':
		print('Evaluating..')
		trainer.evaluate(X, y, config)

	elif config.action == 'plot_calibration':
		print('Plotting Calibration..')
		config.plot_name = 'calibration_plots/calibration_final/{}.png'.format(config.dataset)
		# config.plot_name = 'calibration_plots/alzheimers/{}.png'.format(config.dataset)
		# config.plot_name = 'calibration_plots/alzheimers/{}.png'.format(config.model_dir.split('/')[-1])

		experiments.plot_calibration(X, y, config)

	elif config.action == 'toy_regression':
		print('Toy regression ..')
		experiments.plot_toy_regression(config)

	elif config.action == 'plot_ood':
		print('Plotting OOD..')
		# config.plot_name = 'entropy_plots/{}/'.format(config.dataset)
		config.plot_name = 'entropy_plots_human/{}/'.format(config.dataset)
		experiments.plot_ood(X, y, config)

	elif config.action == 'plot_kl':
		print('Plotting KL..')
		config.plot_name = 'kl_plots/{}.png'.format(config.dataset)

		experiments.plot_kl(X, y, config)

	elif config.action == 'show':
		print('Showing..')

		experiments.show(X, y, config)

run_all = False
if __name__ == '__main__':
	if run_all:
		datasets = ['boston', 'cement', 'power_plant', 'wine', 'yacht', 'kin8nm', 'energy_efficiency', 'protein']
		for d in datasets:

			config.dataset = d
			config.model_dir = 'deepmind-1000/{}-1000'.format(d)
			main(config)
	else:
		main(config)