import utils

import argparse
import os

import numpy as np
import tensorflow as tf

class Opts:
	def __init__(self):
		self.parser = argparse.ArgumentParser()

		self.subparsers = self.parser.add_subparsers(help='train | evaluate | experiment', dest='task')

		# Train Task
		self.parser_train = self.subparsers.add_parser('train', help='Train the model')

		self.parser_train.add_argument('--datasets_dir', required=True, help='Path to dataset')
		self.parser_train.add_argument('--model_dir', required=True, help='Path to save')
		self.parser_train.add_argument('--dataset', required=True, help='One of 11 datasets to use')

		self.parser_train.add_argument('--n_folds', default=20, type=int, help='n folds to cross-validate')
		self.parser_train.add_argument('--n_models', default=5, type=int, help='n models in ensemble')
		self.parser_train.add_argument('--lr', default=1e-1, type=float, help='learning rate')
		self.parser_train.add_argument('--epochs', default=1000, type=int, help='epochs')
		self.parser_train.add_argument('--batch_size', default=100, type=int, help='batch size')

		self.parser_train.add_argument('--build_model', default='combined_pog', help='Type of model to build')
		self.parser_train.add_argument('--units_type', default='prorated', help='Split units proportionately')
		self.parser_train.add_argument('--mod_split', default='computation_split', help='computation_split | human | none')
		self.parser_train.add_argument('--mixture_approximation', default='gaussian', help='gaussian | none')

		self.parser_train.add_argument('--verbose', type=int, default=1)


		# Evaluate Task
		self.parser_evaluate = self.subparsers.add_parser('evaluate', help='Evaluate the model')

		self.parser_evaluate.add_argument('--datasets_dir', required=True, help='Path to dataset')
		self.parser_evaluate.add_argument('--model_dir', required=True, help='Path to load models')
		self.parser_evaluate.add_argument('--dataset', required=True, help='One of 11 datasets to use')


		self.parser_evaluate.add_argument('--n_folds', default=20, type=int, help='n folds to cross-validate')
		self.parser_evaluate.add_argument('--n_models', default=5, type=int, help='n models in ensemble')
		self.parser_evaluate.add_argument('--build_model', default='combined_pog', help='Type of model to build')
		self.parser_evaluate.add_argument('--units_type', default='prorated', help='Split units proportionately')
		self.parser_evaluate.add_argument('--mod_split', default='computation_split', help='computation_split | human | none')
		self.parser_evaluate.add_argument('--mixture_approximation', default='gaussian', help='gaussian | none')

		self.parser_evaluate.add_argument('--verbose', default=0)

		# Experiment Task
		self.parser_experiment = self.subparsers.add_parser('experiment', help='Experiments to run on model')

		self.parser_experiment.add_argument('--exp_name', required=True, help='[defer_simulation, toy_regression, \
			clusterwise_ood, kl_mode, show_summary, empirical_rule_test]')

		self.parser_experiment.add_argument('--datasets_dir', required=True, help='Path to dataset')
		self.parser_experiment.add_argument('--model_dir', required=True, help='Path to load models')
		self.parser_experiment.add_argument('--dataset', required=True, help='One of 11 datasets to use')
		self.parser_experiment.add_argument('--plot_path', required=False, default='plots', help='Plot path')


		self.parser_experiment.add_argument('--n_folds', default=20, type=int, help='n folds to cross-validate')
		self.parser_experiment.add_argument('--n_models', default=5, type=int, help='n models in ensemble')
		self.parser_experiment.add_argument('--build_model', default='combined_pog', help='Type of model to build')
		self.parser_experiment.add_argument('--units_type', default='prorated', help='Split units proportionately')
		self.parser_experiment.add_argument('--mod_split', default='computation_split', help='computation_split | human | none')
		self.parser_experiment.add_argument('--mixture_approximation', default='gaussian', help='gaussian | none')
		self.parser_experiment.add_argument('--power', default=3, type=int, help='Only for toy regression')

		self.parser_experiment.add_argument('--verbose', type=int, default=1)

	def parse(self):
		config = self.parser.parse_args()

		# Quick Access
		# config = utils.EasyDict({
		# 	'model_dir' : 'deepmind-1000/cement-1000',
		# 	# 'model_dir' : 'temp',
		# 	# 'model_dir' : 'alzheimers/alzheimers_models/alzheimers-5000-e-3-bs-16',
		# 	# 'model_dir' : 'human/power_plant-1000',
		# 	# 'model_dir': 'toys/toy2d',

		# 	'datasets_dir': 'datasets',
		# 	'dataset': 'cement',

		# 	# 'task': 'train',
		# 	# 'task': 'evaluate',
		# 	'task': 'experiment',

		# 	# 'exp_name': 'defer_simulation',
		# 	'exp_name': 'clusterwise_ood',
		# 	# 'exp_name': 'kl_mode',
		# 	# 'exp_name': 'show_summary',
		# 	# 'exp_name': 'toy_regression',
		# 	# 'exp_name': 'empirical_rule_test',

		# 	'build_model': 'combined_pog',
		# 	# 'build_model': 'combined_multivariate',
		# 	# 'build_model': 'gaussian',

		# 	'mod_split': 'computation_split',
		# 	# 'mod_split': 'human',
		# 	# 'mod_split': 'none',

		# 	'n_folds': 20,
		# 	'n_models' :  5,

		# 	'units_type': 'absolute',
		# 	# 'units_type': 'prorated',

		# 	'mixture_approximation': 'gaussian',
		# 	# 'mixture_approximation': 'none',

		# 	'plot_path': 'plots/clusterwise_ood',

		# 	'verbose': 0,

		# 	'lr': 0.01,
		# 	'epochs': 1500,
		# 	'batch_size': 8
		# 	})

		return config

