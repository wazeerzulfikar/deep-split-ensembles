import os
from pathlib import Path
import numpy as np

from evaluator import evaluate
import combined_uncertainty
import load_dataset

# Config to choose the hyperparameters for everything
class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name] 

config = EasyDict({
	'model_dir' : 'deepmind/boston',
	'regression_datasets_dir': 'datasets',
	'dataset': 'boston',

	# 'action': 'train',
	'action': 'evaluate',
	# 'action': 'plot',

	'build_model': 'combined',
	'mod_split': 'computation_split',
	'n_folds': 20,
	'n_models' :  5,

	'units_type': 'absolute',
	# 'units_type': 'prorated',

	'plot_name': 'calibration_plots/boston-1000.png',

	'verbose': 0,

	'lr': 0.1,
	'epochs': 1000,
	'batch_size': 100
	})

def main(config):
	data = load_dataset.load_dataset(config)
	n_feature_sets = len(data.keys()) - 1
	X = [np.array(data['{}'.format(i)]) for i in range(n_feature_sets)]
	y = np.array(data['y'])

	config.n_feature_sets = n_feature_sets
	config.feature_split_lengths = [i.shape[1] for i in X]

	print('Number of feature sets ', n_feature_sets)
	[print('Shape of feature set {} {}'.format(e, np.array(i).shape)) for e,i in enumerate(X)]

	model_dir = Path(config.model_dir)
	model_dir.mkdir(parents=True, exist_ok=True)

	if config.dataset in ['boston', 'cement', 'power_plant', 'wine', 'yacht', 'kin8nm', 'energy_efficiency', 'naval']:
		config.units = 50
	elif config.dataset in ['MSD', 'protein']:
		config.units = 100

	if config.action == 'train':
		print('Training..')
		combined_uncertainty.train(X, y, config)

	elif config.action == 'evaluate':
		print('Evaluatig..')
		combined_uncertainty.evaluate(X, y, config)

	elif config.action == 'plot':
		print('Plotting..')
		combined_uncertainty.plot(X, y, config)

if __name__ == '__main__':
	main(config)