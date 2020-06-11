from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Config to choose the hyperparameters for everything
class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name] 

def standard_scale(x_train, x_test):
	scalar = StandardScaler()
	scalar.fit(x_train)
	x_train = scalar.transform(x_train)
	x_test = scalar.transform(x_test)
	return x_train, x_test

def make_model_dir(model_dir):
	model_dir = Path(model_dir)
	model_dir.mkdir(parents=True, exist_ok=True)

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