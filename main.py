import os
import numpy as np
np.random.seed(0)

import trainer
import dataset
import experiments
import utils
from opts import Opts

def main(config):

	if config.mod_split == 'computation_split' and config.dataset in ['boston', 'wine', 'kin8nm', 'naval', 'protein']:
		config.hc_threshold = 0.5
	elif config.mod_split == 'computation_split' and config.dataset in ['cement', 'energy_efficiency', 'power_plant', 'yacht']:
		config.hc_threshold = 0.75

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

	if config.build_model == 'mc_dropout':
		config.n_models = 1

	if config.dataset in ['boston', 'cement', 'power_plant', 'wine', 'yacht', 'kin8nm', 'energy_efficiency', 'naval', 'life']:
		config.units = 50
	elif config.dataset in ['msd', 'protein', 'toy']:
		config.units = 100
	elif config.dataset in ['alzheimers', 'alzheimers_test']:
		config.units = 100
		config.feature_split_lengths[-1] = 21 # COMPARE features after PCA
		config.n_folds = 5

	if config.build_model == 'combined_pog' and config.dataset in ['cement', 'protein', 'yacht', 'power_plant']:
		config.y_scaling = 1

	if config.dataset == 'protein':
		config.n_folds = 5

	if config.dataset == 'msd':
		config.n_models = 2

	if config.mod_split == 'none':
		config.n_feature_sets = 1

	if config.task == 'train':
		print('Training..')
		trainer.train(X, y, config)

	elif config.task == 'evaluate':
		print('Evaluating..')
		trainer.evaluate(X, y, config)

	elif config.task == 'experiment':

		config.plot_name = os.path.join(config.plot_path, '{}_{}.png'.format(config.dataset, config.exp_name))
		
		if config.exp_name == 'defer_simulation':
			print('Plotting Calibration..')
			experiments.plot_defer_simulation(X, y, config)

		elif config.exp_name == 'toy_regression':
			print('Toy regression ..')
			experiments.plot_toy_regression(config)

		elif config.exp_name == 'clusterwise_ood':
			print('Plotting OOD..')
			experiments.plot_ood(X, y, config)

		elif config.exp_name == 'kl_mode':
			print('Plotting KL..')
			experiments.plot_kl(X, y, config)

		elif config.exp_name == 'show_summary':
			print('Showing..')
			experiments.show_model_summary(X, y, config)

		elif config.exp_name == 'empirical_rule_test':
			print('Emprical rule tests..')
			experiments.empirical_rule_test(X, y, config)

if __name__ == '__main__':
	opts = Opts()
	config = opts.parse()
	config.verbose = (int)(config.verbose)
	main(config)


# python main.py train --datasets_dir datasets --dataset boston --model_dir boston_anc_models_comsplit_10folds --n_folds 3 --build_model anc_ens --verbose 1