from models import build_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os


def standard_scale(x_train, x_test):
	scalar = StandardScaler()
	scalar.fit(x_train)
	x_train = scalar.transform(x_train)
	x_test = scalar.transform(x_test)
	return x_train, x_test

def evaluate(config, data):

	final_train_score, final_val_score = [], []
	final_feature_models_train_score, final_feature_models_val_score = [], []

	y = data['y']
	kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
	fold=1
	for train_index, test_index in kf.split(y):
	### starts here
		print('~ Fold {} starting ~'.format(fold))
		feature_sets = len(data)-1
		feature_models_train_preds, feature_models_val_preds = [], []
		feature_models_train_score, feature_models_val_score = [], []
		for feature_set in data:
			if feature_set=='y':
				continue
			else:
				pass
			X = data[feature_set]
			y = data['y']

			x_train = np.asarray(X[train_index])
			x_val = np.asarray(X[test_index])

			y_train = np.asarray(y[train_index])
			y_val = np.asarray(y[test_index])

			x_train, x_val = standard_scale(x_train, x_val)

			fold_train_score, fold_val_score = [], []
			
			train_preds, val_preds = [0]*len(x_train), [0]*len(x_val)

			for model_number in range(config.n_models):
				
				if config.build_model=='point':
					model, history = train_a_fold(fold, model_number+1, config, x_train, y_train, x_val, y_val)

					train_preds += model.predict(x_train)[:, 0]
					val_preds += model.predict(x_val)[:, 0]

			if config.build_model=='point':
				train_preds /= config.n_models
				val_preds /= config.n_models

				feature_models_train_preds.append(np.array(train_preds))
				feature_models_val_preds.append(np.array(val_preds))

				feature_models_train_score.append(mean_squared_error(y_train, np.array(train_preds), squared=False))
				feature_models_val_score.append(mean_squared_error(y_val, np.array(val_preds), squared=False))

		final_feature_models_train_score.append(np.array(feature_models_train_score))
		final_feature_models_val_score.append(np.array(feature_models_val_score))

		final_train_preds = np.mean(np.array(feature_models_train_preds), axis=0)
		final_val_preds = np.mean(np.array(feature_models_val_preds), axis=0)
		final_train_score.append(mean_squared_error(y_train, final_train_preds, squared=False))
		final_val_score.append(mean_squared_error(y_val, final_val_preds, squared=False))
		fold+=1
	# print("\nBest epochs : ", best_epochs)
	if config.build_model=='point':
		print("Featurewise Models Train RMSE:", ['{:.3f} +/- {:.3f}'.format(i, j) for i, j in zip(np.mean(final_feature_models_train_score, axis=0), np.std(final_feature_models_train_score, axis=0))])
		print("Featurewise Models Val RMSE:", ['{:.3f} +/- {:.3f}'.format(i, j) for i, j in zip(np.mean(final_feature_models_val_score, axis=0), np.std(final_feature_models_val_score, axis=0))])
		print("Ensemble Train RMSE: {:.3f} +/- {:.3f}".format(np.mean(final_train_score), np.std(final_train_score)))
		print("Ensemble Val RMSE: {:.3f} +/- {:.3f}".format(np.mean(final_val_score), np.std(final_val_score)))

	### ends here

def train_a_fold(fold, model_number, config, x_train, y_train, x_val, y_val):
	
	model, loss = build_model(config)

	epochs = config.epochs
	lr = config.learning_rate
	batch_size = config.batch_size

	model.compile(loss=loss, optimizer=Adam(learning_rate=lr))
	checkpoint_filepath = os.path.join(config.model_dir, config.dataset, config.expt_name, 'fold_{}_model_{}.h5'.format(fold, model_number))
	checkpoints = ModelCheckpoint(checkpoint_filepath,
								  monitor='val_loss', 
								  verbose=0, 
								  save_best_only=True,
								  save_weights_only=False,
								  mode='auto',
								  save_freq='epoch')

	history = model.fit(x_train, y_train,
				  epochs=epochs,
				  batch_size=batch_size,
				  verbose=1,
				  callbacks=[checkpoints],
				  validation_data=(x_val, y_val))

	if config.build_model=='point':
		model = load_model(checkpoint_filepath)
	elif config.build_model=='gaussian':
		model.load_weights(checkpoint_filepath)

	return model, history