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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def standard_scale(x_train, x_test):
	scalar = StandardScaler()
	scalar.fit(x_train)
	x_train = scalar.transform(x_train)
	x_test = scalar.transform(x_test)
	return x_train, x_test

def evaluate(config, data):

	final_train_score, final_val_score = [], []
	final_train_rmse, final_val_rmse = [], []
	final_feature_models_train_score, final_feature_models_val_score = [], []

	y = data['y']
	kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
	fold=1
	for train_index, test_index in kf.split(y):
		print('~ Fold {} starting ~'.format(fold))

		feature_sets = len(data)-1
		feature_models_train_preds, feature_models_val_preds = [], []
		feature_models_train_score, feature_models_val_score = [], []
		feature_keys = sorted([i for i in range(feature_sets)])

		if config.dataset=='msd':
			train_index = [x for x in range(463715)]
			test_index = [x for x in range(463715, 515345)]
		
		for feature_set in feature_keys:
			if feature_set=='y':
				continue
			else:
				pass
			X = data[str(feature_set)]
			y = data['y']

			x_train = np.asarray(X[train_index])
			x_val = np.asarray(X[test_index])

			y_train = np.asarray(y[train_index])
			y_val = np.asarray(y[test_index])

			x_train, x_val = standard_scale(x_train, x_val)

			fold_train_score, fold_val_score = [], []
			
			train_preds, val_preds = [0]*len(x_train), [0]*len(x_val)

			mus_train, mus_val = [], []
			sigmas_train, sigmas_val = [], []

			for model_number in range(config.n_models):
				model, history = train_a_fold(fold, model_number+1, config, x_train, y_train, x_val, y_val)

				if config.build_model=='point':					
					train_preds += model.predict(x_train)[:, 0]
					val_preds += model.predict(x_val)[:, 0]

				if config.build_model=='gaussian':
					pred_train = model(x_train)
					mu_train = pred_train.mean()
					sigma_train = pred_train.stddev()
					mus_train.append(mu_train.numpy())
					sigmas_train.append(sigma_train.numpy())

					pred_val = model(x_val)
					mu_val = pred_val.mean()
					sigma_val = pred_val.stddev()
					mus_val.append(mu_val.numpy())
					sigmas_val.append(sigma_val.numpy())

					val_score = np.min(history.history['val_loss'])
					train_score = history.history['loss'][np.argmin(history.history['val_loss'])]
				
			if config.build_model=='point':
				train_preds /= config.n_models
				val_preds /= config.n_models

				feature_models_train_preds.append(np.array(train_preds))
				feature_models_val_preds.append(np.array(val_preds))

				feature_models_train_score.append(mean_squared_error(y_train, np.array(train_preds), squared=False))
				feature_models_val_score.append(mean_squared_error(y_val, np.array(val_preds), squared=False))

		if config.build_model=='point':
			final_feature_models_train_score.append(np.array(feature_models_train_score))
			final_feature_models_val_score.append(np.array(feature_models_val_score))

			final_train_preds = np.mean(np.array(feature_models_train_preds), axis=0)
			final_val_preds = np.mean(np.array(feature_models_val_preds), axis=0)
			final_train_score.append(mean_squared_error(y_train, final_train_preds, squared=False))
			final_val_score.append(mean_squared_error(y_val, final_val_preds, squared=False))

		if config.build_model=='gaussian':

			mus_train, sigmas_train = np.concatenate(mus_train, axis=-1), np.concatenate(sigmas_train, axis=-1)
			ensemble_mu_train = np.mean(mus_train, axis=-1).reshape(-1,1)
			ensemble_sigma_train = np.sqrt(np.mean(np.square(sigmas_train) + np.square(mus_train), axis=-1).reshape(-1,1) - np.square(ensemble_mu_train))
		
			mus_val, sigmas_val = np.concatenate(mus_val, axis=-1), np.concatenate(sigmas_val, axis=-1)
			ensemble_mu_val = np.mean(mus_val, axis=-1).reshape(-1,1)
			ensemble_sigma_val = np.sqrt(np.mean(np.square(sigmas_val) + np.square(mus_val), axis=-1).reshape(-1,1) - np.square(ensemble_mu_val))
		
			tfd = tfp.distributions
			ensemble_dist_train = tfd.Normal(loc=ensemble_mu_train, scale=ensemble_sigma_train)
			ensemble_dist_val = tfd.Normal(loc=ensemble_mu_val, scale=ensemble_sigma_val)

			ensemble_true_train_log_probs = ensemble_dist_train.log_prob(y_train).numpy()
			final_train_score.append(np.mean(-ensemble_true_train_log_probs))
			ensemble_true_val_log_probs = ensemble_dist_val.log_prob(y_val).numpy()
			final_val_score.append(np.mean(-ensemble_true_val_log_probs))
			final_train_rmse.append(mean_squared_error(y_train, ensemble_mu_train, squared=False))
			final_val_rmse.append(mean_squared_error(y_val, ensemble_mu_val, squared=False))

		if config.dataset=='msd':
			break

		fold+=1

	if config.build_model=='point':
		print("Featurewise Models Train RMSE:", ['{:.3f} +/- {:.3f}'.format(i, j) for i, j in zip(np.mean(final_feature_models_train_score, axis=0), np.std(final_feature_models_train_score, axis=0))])
		print("Featurewise Models Val RMSE:", ['{:.3f} +/- {:.3f}'.format(i, j) for i, j in zip(np.mean(final_feature_models_val_score, axis=0), np.std(final_feature_models_val_score, axis=0))])
		print("Ensemble Train RMSE: {:.3f} +/- {:.3f}".format(np.mean(final_train_score), np.std(final_train_score)))
		print("Ensemble Val RMSE: {:.3f} +/- {:.3f}".format(np.mean(final_val_score), np.std(final_val_score)))
	
	if config.build_model=='gaussian':
		print("\nTrain NLL : ", final_train_score)
		print("Val NLL : ", final_val_score)
		print("Train NLL mean : ", np.mean(final_train_score), "+/-", np.std(final_train_score))
		print("Val NLL mean : ", np.mean(final_val_score), "+/-", np.std(final_val_score))
		print("\nTrain RMSE : ", final_train_rmse)
		print("Val RMSE : ", final_val_rmse)		
		print("Train RMSE mean : ", np.mean(final_train_rmse), "+/-", np.std(final_train_rmse))
		print("Val RMSE mean : ", np.mean(final_val_rmse), "+/-", np.std(final_val_rmse))


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
				  verbose=0,
				  callbacks=[checkpoints],
				  validation_data=(x_val, y_val))

	if config.build_model=='point':
		model = load_model(checkpoint_filepath)
	elif config.build_model=='gaussian':
		model.load_weights(checkpoint_filepath)

	return model, history
