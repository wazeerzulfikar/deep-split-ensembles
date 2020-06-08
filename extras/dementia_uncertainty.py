import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import *

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import glob, os, math, time
from decimal import *
import numpy as np
np.random.seed(0)
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import dataset
import utils

tfd = tfp.distributions

def create_pause_model():
	model = tf.keras.Sequential()
	model.add(Input(shape=(11,)))
	model.add(BatchNormalization())
	model.add(Dense(16, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(16, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(2, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.1)))
	model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[..., 1:])+1e-6)))
	return model

def create_compare_model():
	model = tf.keras.Sequential()
	model.add(Input(shape=(21,)))
	model.add(Dense(24, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(8, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(8, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(24, activation='relu', activity_regularizer=tf.keras.regularizers.l1(0.1)))
	model.add(Dropout(0.3))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)))
	model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[..., 1:])+1e-6)))
	return model

def create_intervention_model():
	model = tf.keras.Sequential()
	model.add(Input((32,3)))
	model.add(LSTM(12))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(2, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.1)))
	model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[..., 1:])+1e-6)))
	return model

def create_combined_model():
	intervention_inputs = Input((32, 3))
	pause_inputs = Input((11,))
	compare_inputs = Input((21,))

	intervention_x = LSTM(12)(intervention_inputs)
	intervention_x = BatchNormalization()(intervention_x)
	intervention_x = Dropout(0.4)(intervention_x)
	intervention_x = Dense(16, activation='relu')(intervention_x)
	intervention_x = Dense(16, activation='relu')(intervention_x)

	intervention_std = Dense(1)(intervention_x)

	pause_x = BatchNormalization()(pause_inputs)
	pause_x = Dense(16, activation='relu')(pause_x)
	pause_x = BatchNormalization()(pause_x)
	pause_x = Dropout(0.2)(pause_x)
	pause_x = Dense(16, activation='relu')(pause_x)
	pause_x = BatchNormalization()(pause_x)
	pause_x = Dropout(0.5)(pause_x)
	pause_x = Dense(32, activation='relu')(pause_x)
	pause_x = Dense(16, activation='relu')(pause_x)
	pause_x = Dense(8, activation='relu')(pause_x)
	pause_x = Dropout(0.3)(pause_x)

	pause_std = Dense(1)(pause_x)

	compare_x = Dense(24, activation='relu')(compare_inputs)
	compare_x = BatchNormalization()(compare_x)
	compare_x = Dense(8, activation='relu')(compare_x)
	compare_x = BatchNormalization()(compare_x)
	compare_x = Dense(8, activation='relu')(compare_x)
	compare_x = BatchNormalization()(compare_x)
	compare_x = Dense(24, activation='relu', activity_regularizer=tf.keras.regularizers.l1(0.1))(compare_x)
	compare_x = BatchNormalization()(compare_x)
	compare_x = Dropout(0.3)(compare_x)
	compare_x = Dense(8, activation='relu')(compare_x)

	compare_std = Dense(1)(compare_x)

	mu = Concatenate()([intervention_x, pause_x, compare_x])
	# mu = BatchNormalization()(mu)
	mu = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.1))(mu)

	intervention_gaus = Concatenate()([mu, intervention_std])
	pause_gaus = Concatenate()([mu, pause_std])
	compare_gaus = Concatenate()([mu, compare_std])

	intervention_output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6), name='intervention')(intervention_gaus)
	pause_output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6), name='pause')(pause_gaus)
	compare_output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6), name='compare')(compare_gaus)

	return tf.keras.models.Model(inputs=[intervention_inputs, pause_inputs, compare_inputs],
	 outputs=[intervention_output, pause_output, compare_output])

def create_combined_1_model():
	intervention_inputs = Input((32, 3))
	pause_inputs = Input((11,))
	compare_inputs = Input((21,))

	intervention_x = LSTM(24)(intervention_inputs)
	intervention_x = BatchNormalization()(intervention_x)

	intervention_x = Dense(16, activation='relu')(intervention_x)
	intervention_x = BatchNormalization()(intervention_x)
	intervention_x = Dense(8, activation='relu')(intervention_x)
	intervention_x = BatchNormalization()(intervention_x)
	intervention_x = Dropout(0.2)(intervention_x)

	intervention_std = Dense(1)(intervention_x)

	pause_x = Dense(24, activation='relu')(pause_inputs)
	pause_x = BatchNormalization()(pause_x)
	pause_x = Dense(16, activation='relu')(pause_x)
	pause_x = BatchNormalization()(pause_x)
	pause_x = Dense(8, activation='relu')(pause_x)
	pause_x = BatchNormalization()(pause_x)
	pause_x = Dropout(0.2)(pause_x)

	pause_std = Dense(1)(pause_x)

	compare_x = Dense(24, activation='relu')(compare_inputs)
	compare_x = BatchNormalization()(compare_x)
	compare_x = Dense(16, activation='relu')(compare_x)
	compare_x = BatchNormalization()(compare_x)
	compare_x = Dense(8, activation='relu')(compare_x)
	compare_x = BatchNormalization()(compare_x)
	compare_x = Dropout(0.2)(compare_x)

	compare_std = Dense(1)(compare_x)

	mu = Concatenate()([intervention_x, pause_x, compare_x])
	mu = Dense(8, activation='relu')(mu)
	mu = BatchNormalization()(mu)
	mu = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(mu)

	intervention_gaus = Concatenate()([mu, intervention_std])
	pause_gaus = Concatenate()([mu, pause_std])
	compare_gaus = Concatenate()([mu, compare_std])

	intervention_output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6), name='intervention')(intervention_gaus)
	pause_output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6), name='pause')(pause_gaus)
	compare_output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6), name='compare')(compare_gaus)

	return tf.keras.models.Model(inputs=[intervention_inputs, pause_inputs, compare_inputs],
	 outputs=[intervention_output, pause_output, compare_output])

def train_a_fold(
	model_type, 
	x_train, y_train,
	x_val, y_val, 
	model_dir, model_id=1, fold=1):

	if model_type == 'pause':
		model = create_pause_model()
		lr = 0.01
		epochs = 5000

	elif model_type == 'intervention':
		model = create_intervention_model()
		lr = 0.001
		epochs = 2000

	elif 'combined' in model_type:
		model = create_combined_1_model()
		lr = 0.01
		epochs = 2000	

	negloglik = lambda y, p_y: -p_y.log_prob(y)
	custom_mse = lambda y, p_y: tf.keras.losses.mean_squared_error(y, p_y.mean())
	mse_wrapped = utils.MeanMetricWrapper(custom_mse, name='custom_mse')
	# try:
	# 	for i in os.path.join(model_dir, model_type, '*.h5'):		os.remove(i)
	# except:
	# 	pass	

	# checkpoint_filepath = os.path.join(model_dir, model_type, '{epoch:d}-{val_loss:.3f}.h5')
	checkpoint_filepath = os.path.join(model_dir, model_type, 'model_nll_{}.h5'.format(model_id))
	checkpointer = tf.keras.callbacks.ModelCheckpoint(
			checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
			save_weights_only=True, mode='auto', save_freq='epoch')

	# early_stopping = tf.keras.callbacks.EarlyStopping(
	# 			monitor='val_loss', min_delta=0.001, patience=1000, verbose=0, mode='auto',
	# 			baseline=None, restore_best_weights=True
	# 		)

	model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
				  loss=[negloglik, negloglik, negloglik])
				  # metrics=[mse_wrapped, mse_wrapped, mse_wrapped])
				  # metrics=[custom_mse])

	hist = model.fit(x_train, [y_train, y_train, y_train],
					batch_size=8,
					epochs=epochs,
					verbose=1,
					callbacks=[checkpointer],
					validation_data=(x_val, [y_val, y_val, y_val]))
	# print(model.summary())
	epoch_val_losses = hist.history['val_loss']
	best_epoch_val_loss, best_epoch = np.min(epoch_val_losses), np.argmin(epoch_val_losses)+1
	best_epoch_train_loss = hist.history['loss'][best_epoch-1]
	# checkpoint_filepath = os.path.join(model_dir, model_type, '{:d}-{:.3f}.h5'.format(best_epoch, best_epoch_val_loss))
	model.load_weights(checkpoint_filepath)

	return model, best_epoch_train_loss, best_epoch_val_loss, best_epoch

config = utils.EasyDict({
	'task': 'classification',
	# 'task': 'regression',

	# 'dataset_dir': '../DementiaBank'
	'dataset_dir': '../ADReSS-IS2020-data/train',

	'model_dir': 'models/bagging',
	'model_types': ['intervention', 'pause', 'compare'],

	'training_type': 'bagging',
	# 'training_type' :'boosting',

	'n_folds': 5,

	# 'dataset_split' :'full_dataset',
	'dataset_split' :'k_fold',

	'voting_type': 'hard_voting',
	# 'voting_type': 'soft_voting',
	# 'voting_type': 'learnt_voting',


	'longest_speaker_length': 32,
	'n_pause_features': 11,
	'compare_features_size': 21,
	'split_reference': 'samples'
})

data = dataset.prepare_data(config)

results = {}


model_types = ['pause'] # from scratch running in tmux 2
model_types = ['intervention']
model_types = ['combined_1']
model_types = ['combined_deep_ensemble']

model_dir = 'uncertainty'
model_dir = Path(model_dir)
model_dir.joinpath(model_types[0]).mkdir(parents=True, exist_ok=True)
model_dir = str(model_dir)

model_type = model_types[0]
x_inv, x_pause, x_compare, y = data['intervention'], data['pause'], data['compare'], data['y_reg']
n_samples = x_inv.shape[0]


n_models = 10
mus, inv_sigmas, pause_sigmas, compare_sigmas = [], [], [], []
train_nlls, val_nlls = [], []
start = time.time()
for model_id in range(n_models):
	print('~ Training model {} ~'.format(model_id+1))
	# p = np.random.permutation(n_samples) # mus are corresponding to different samples, cant mean it out
	# x, y = x[p], y[p]

	x_train_i, x_val_i = x_inv[int(0.2*n_samples):], x_inv[:int(0.2*n_samples)]
	x_train_p, x_val_p = x_pause[int(0.2*n_samples):], x_pause[:int(0.2*n_samples)]
	x_train_c, x_val_c = x_compare[int(0.2*n_samples):], x_compare[:int(0.2*n_samples)]

	x_train_c, x_val_c = utils.normalize_compare_features(x_train_c, x_val_c, config.compare_features_size)

	y_train, y_val = y[int(0.2*n_samples):], y[:int(0.2*n_samples)]
	n_val_samples = x_val_i.shape[0]

	# model, best_epoch_train_loss, best_epoch_val_loss, best_epoch = train_a_fold(
	# 	model_type, 
	# 	[x_train_i, x_train_p, x_train_c], y_train,
	# 	[x_val_i, x_val_p, x_val_c], y_val, model_dir, model_id=model_id)

	# train_nll, val_nll = best_epoch_train_loss, best_epoch_val_loss

	# print('Best Epoch: {:d}'.format(best_epoch))
	# print('Train NLL: {:.3f}'.format(best_epoch_train_loss)) 
	# print('Val NLL: {:.3f}'.format(best_epoch_val_loss)) # NLL
	# train_nlls.append(train_nll)
	# val_nlls.append(val_nll)
	########### UNCOMMENT FOR EVALUATION ###########
	model = create_combined_model()

	checkpoint_filepath = os.path.join(model_dir, model_type, 'model_nll_{}.h5'.format(model_id)) ########### CHANGE WHICH MODEL TO LOAD nll OR mse
	model.load_weights(checkpoint_filepath)
	########### UNCOMMENT FOR EVALUATION ###########


	y_val = y_val.reshape(-1,1)
	pred = model([x_val_i, x_val_p, x_val_c])
	inv_pred, pause_pred, compare_pred = pred[0], pred[1], pred[2]
	
	mu = inv_pred.mean()
	inv_sigma = inv_pred.stddev()

	assert np.all(mu.numpy() == pause_pred.mean().numpy()), 'uhoh'
	pause_sigma = pause_pred.stddev()
	compare_sigma = compare_pred.stddev()

	mus.append(mu.numpy())
	inv_sigmas.append(inv_sigma.numpy())
	pause_sigmas.append(pause_sigma.numpy())
	compare_sigmas.append(compare_sigma.numpy())


	val_rmse = mean_squared_error(y_val, mu, squared=False)

	print()
	print(model_type)

	print('Val RMSE: {:.3f}'.format(val_rmse))

	inv_pred_probs = inv_pred.prob(mu)
	pause_pred_probs = pause_pred.prob(mu)
	compare_pred_probs = compare_pred.prob(mu)

	inv_true_y_probs = inv_pred.prob(y_val)
	pause_true_y_probs = pause_pred.prob(y_val)
	compare_true_y_probs = compare_pred.prob(y_val)

	inv_pred_log_probs = inv_pred.log_prob(mu)
	pause_pred_log_probs = pause_pred.log_prob(mu)
	compare_pred_log_probs = compare_pred.log_prob(mu)

	inv_true_y_log_probs = inv_pred.log_prob(y_val)
	pause_true_y_log_probs = pause_pred.log_prob(y_val)
	compare_true_y_log_probs = compare_pred.log_prob(y_val)

	inv_std = inv_pred.stddev()
	pause_std = pause_pred.stddev()
	compare_std = compare_pred.stddev()

	inv_log_std = np.log(inv_std)
	pause_log_std = np.log(pause_std)
	compare_log_std = np.log(compare_std)

	inv_entropy = inv_pred.entropy()
	pause_entropy = pause_pred.entropy()
	compare_entropy = compare_pred.entropy()

	print()
	for i in range(n_val_samples):
		# tf.print('Pred: {:.3f}'.format(mu[i][0]), '\t\t\tProb: {:.7f}'.format(pred_probs[i][0]), '\t\t\tTrue: {}'.format(y_val[i][0]), '\t\t\tProb: {:.7f}'.format(true_y_probs[i][0]), '\t\t\tStd Dev: {:.7f}'.format(sigma[i][0]), '\t\t\tEntropy: {:.7f}'.format(pred.entropy()[i][0]))
		tf.print('Pred: {:.3f}'.format(mu[i][0]), 
			 '\tTrue: {}'.format(y_val[i][0]), '\tInv T Prob: {:.7f}'.format(inv_true_y_probs[i][0]), '\tPause T Prob: {:.7f}'.format(pause_true_y_probs[i][0]), '\tCompare T Prob: {:.7f}'.format(compare_true_y_probs[i][0]),
			 '\tInv Std Dev: {:.7f}'.format(inv_sigma[i][0]), '\tPause Std Dev: {:.7f}'.format(pause_sigma[i][0]), '\tCompare Std Dev: {:.7f}'.format(compare_sigma[i][0]))
	print()
	# for i in range(n_val_samples):
	# 	# tf.print('Pred: {:.3f}'.format(mu[i][0]), '\t\t\tProb: {:.7f}'.format(pred_probs[i][0]), '\t\t\tTrue: {}'.format(y_val[i][0]), '\t\t\tProb: {:.7f}'.format(true_y_probs[i][0]), '\t\t\tStd Dev: {:.7f}'.format(sigma[i][0]), '\t\t\tEntropy: {:.7f}'.format(pred.entropy()[i][0]))
	# 	tf.print('Pred: {:.3f}'.format(mu[i][0]), '\t\tPause Prob: {:.7f}'.format(pause_pred_probs[i][0]), '\t\tPause True: {}'.format(y_val[i][0]), '\t\tPause T Prob: {:.7f}'.format(pause_true_y_probs[i][0]), '\t\tPause Std Dev: {:.7f}'.format(pause_sigma[i][0]))
	# print()
	print(model_type)

# exit()

mus =  np.concatenate(mus, axis=-1),
inv_sigmas = np.concatenate(inv_sigmas, axis=-1)
pause_sigmas = np.concatenate(pause_sigmas, axis=-1)
compare_sigmas = np.concatenate(compare_sigmas, axis=-1)

ensemble_mu = np.mean(mus, axis=-1).reshape(-1,1)
ensemble_sigma_inv = np.sqrt(np.mean(np.square(inv_sigmas) + np.square(mus), axis=-1).reshape(-1,1) - np.square(ensemble_mu))
ensemble_sigma_pause = np.sqrt(np.mean(np.square(pause_sigmas) + np.square(mus), axis=-1).reshape(-1,1) - np.square(ensemble_mu))
ensemble_sigma_compare = np.sqrt(np.mean(np.square(compare_sigmas) + np.square(mus), axis=-1).reshape(-1,1) - np.square(ensemble_mu))

ensemble_dist_inv = tfd.Normal(loc=ensemble_mu, scale=ensemble_sigma_inv)
ensemble_dist_pause = tfd.Normal(loc=ensemble_mu, scale=ensemble_sigma_pause)
ensemble_dist_compare = tfd.Normal(loc=ensemble_mu, scale=ensemble_sigma_compare)
# ensemble_pred_probs = ensemble_dist.prob(ensemble_mu).numpy()
# ensemble_pred_log_probs = ensemble_dist_inv.log_prob(ensemble_mu).numpy()
# ensemble_pred_log_probs = ensemble_dist_pause.log_prob(ensemble_mu).numpy()
# ensemble_pred_log_probs = ensemble_dist_compare.log_prob(ensemble_mu).numpy()

ensemble_true_probs_inv = ensemble_dist_inv.prob(y_val).numpy()
ensemble_true_probs_pause = ensemble_dist_pause.prob(y_val).numpy()
ensemble_true_probs_compare = ensemble_dist_compare.prob(y_val).numpy()

ensemble_true_log_probs_inv = ensemble_dist_inv.log_prob(y_val).numpy()
ensemble_true_log_probs_pause = ensemble_dist_pause.log_prob(y_val).numpy()
ensemble_true_log_probs_compare = ensemble_dist_compare.log_prob(y_val).numpy()

ensemble_entropy_inv = ensemble_dist_inv.entropy()
ensemble_entropy_pause = ensemble_dist_pause.entropy()
ensemble_entropy_comare = ensemble_dist_compare.entropy()

ensemble_nll_inv = np.mean(-ensemble_true_log_probs_inv)
ensemble_nll_pause = np.mean(-ensemble_true_log_probs_pause)
ensemble_nll_compare = np.mean(-ensemble_true_log_probs_compare)

print()
print('Deep ensemble results')
for i in range(n_val_samples):
	print('Pred: {:.3f}'.format(ensemble_mu[i][0]), 
			 '\tTrue: {}'.format(y_val[i][0]), '\tInv T Prob: {:.7f}'.format(ensemble_true_probs_inv[i][0]), '\tPause T Prob: {:.7f}'.format(ensemble_true_probs_pause[i][0]), '\tCompare T Prob: {:.7f}'.format(ensemble_true_probs_compare[i][0]),
			 '\tInv Std Dev: {:.7f}'.format(ensemble_sigma_inv[i][0]), '\tPause Std Dev: {:.7f}'.format(ensemble_sigma_pause[i][0]), '\tCompare Std Dev: {:.7f}'.format(ensemble_sigma_compare[i][0]))
	
print()

print('Ensemble inv nll', ensemble_nll_inv)
print('Ensemble pause nll', ensemble_nll_pause)
print('Ensemble compare nll', ensemble_nll_compare)

errors = np.absolute(ensemble_mu - y_val)
print('Ensemble Val RMSE ', mean_squared_error(y_val, ensemble_mu, squared=False))

print('outlier', np.argmax(ensemble_sigma_inv))
print('outlier', np.argmax(ensemble_sigma_pause))
print('outlier', np.argmax(ensemble_sigma_compare))

# ensemble_sigma_inv = np.delete(ensemble_sigma_inv, 19)
# ensemble_sigma_pause = np.delete(ensemble_sigma_pause, 19)
# ensemble_sigma_compare = np.delete(ensemble_sigma_compare, 19)
# stddev_ensemble_mu = np.delete(ensemble_mu, 19)
# stddev_errors = np.delete(errors, 19)


# stddev_errors = errors


# plt.subplot(3, 2, 1)
# # plt.scatter(inv_pred_probs, errors, label='inv', s=2)
# # plt.scatter(pause_pred_probs, errors, label='pause', s=2)
# # plt.scatter(compare_pred_probs, errors, label='compare', s=2)
# # plt.legend(loc='upper right')
# # plt.title('Absolute errors vs pred probs')

# plt.scatter(np.log(ensemble_sigma_inv), stddev_errors, label='inv', s=2)
# plt.scatter(np.log(ensemble_sigma_pause), stddev_errors, label='pause', s=2)
# plt.scatter(np.log(ensemble_sigma_compare), stddev_errors, label='compare', s=2)
# # plt.legend(loc='upper right')
# plt.title('Absolute errors vs log std dev')


# plt.subplot(3, 2, 2)
# multiplier = 10.0
# plt.scatter(ensemble_sigma_inv*multiplier, stddev_errors, label='inv', s=2)
# plt.scatter(ensemble_sigma_pause*multiplier, stddev_errors, label='pause', s=2)
# plt.scatter(ensemble_sigma_compare*multiplier, stddev_errors, label='compare', s=2)
# plt.plot(np.unique(ensemble_sigma_inv*multiplier), np.poly1d(np.polyfit(ensemble_sigma_inv*multiplier, stddev_errors, 1))(np.unique(ensemble_sigma_inv*multiplier)))
# plt.plot(np.unique(ensemble_sigma_pause*multiplier), np.poly1d(np.polyfit(ensemble_sigma_pause*multiplier, stddev_errors, 1))(np.unique(ensemble_sigma_pause*multiplier)))
# plt.plot(np.unique(ensemble_sigma_compare*multiplier), np.poly1d(np.polyfit(ensemble_sigma_compare*multiplier, stddev_errors, 1))(np.unique(ensemble_sigma_compare*multiplier)))

# # plt.legend(loc='upper right')
# plt.title('Absolute errors vs val std dev / 10.0')

# plt.subplot(3, 2, 3)
# plt.scatter(ensemble_sigma_inv, stddev_ensemble_mu, label='inv', s=2)
# # plt.scatter(ensemble_sigma_pause, stddev_ensemble_mu, label='pause', s=2)
# # plt.scatter(ensemble_sigma_compare, stddev_ensemble_mu, label='compare', s=2)
# for i, e in enumerate(stddev_errors):
#     plt.annotate('{:0.2f}'.format(e), (ensemble_sigma_inv[i], stddev_ensemble_mu[i]))
# plt.legend(loc='upper right')
# plt.title('ensemble_mu vs stddev_')

# plt.subplot(3, 2, 4)
# plt.scatter(ensemble_sigma_inv*1.5, stddev_errors, label='inv', s=2)
# plt.scatter(ensemble_sigma_pause*1.5, stddev_errors, label='pause', s=2)
# plt.scatter(ensemble_sigma_compare*1.5, stddev_errors, label='compare', s=2)
# # plt.legend(loc='upper right')
# plt.title('Absolute errors vs val std dev / 1.5')

# plt.subplot(3, 2, 5)
# plt.scatter(ensemble_sigma_inv, stddev_errors, label='inv', s=2)
# plt.scatter(ensemble_sigma_pause, stddev_errors, label='pause', s=2)
# plt.scatter(ensemble_sigma_compare, stddev_errors, label='compare', s=2)

# plt.plot(np.unique(ensemble_sigma_inv), np.poly1d(np.polyfit(ensemble_sigma_inv, stddev_errors, 1))(np.unique(ensemble_sigma_inv)))
# plt.plot(np.unique(ensemble_sigma_pause), np.poly1d(np.polyfit(ensemble_sigma_pause, stddev_errors, 1))(np.unique(ensemble_sigma_pause)))
# plt.plot(np.unique(ensemble_sigma_compare), np.poly1d(np.polyfit(ensemble_sigma_compare, stddev_errors, 1))(np.unique(ensemble_sigma_compare)))

# # plt.legend(loc='upper right')
# plt.title('Absolute errors vs val Std Dev')

# plt.subplot(3, 2, 6)
# plt.scatter(ensemble_entropy_inv, errors, label='inv', s=2)
# plt.scatter(ensemble_entropy_pause, errors, label='pause', s=2)
# plt.scatter(ensemble_entropy_comare, errors, label='compare', s=2)
# # plt.legend(loc='upper right')
# plt.title('Absolute errors vs val Entropy')
# plt.savefig('ensemble_combined.png')

# plt.show()
# exit()

# errors = np.absolute(mus - y_val)
# plt.subplot(3, 2, 1)
# # plt.scatter(inv_pred_probs, errors, label='inv', s=2)
# # plt.scatter(pause_pred_probs, errors, label='pause', s=2)
# # plt.scatter(compare_pred_probs, errors, label='compare', s=2)
# # plt.legend(loc='upper right')
# # plt.title('Absolute errors vs pred probs')

# plt.scatter(inv_log_std, errors, label='inv', s=2)
# plt.scatter(pause_log_std, errors, label='pause', s=2)
# plt.scatter(compare_log_std, errors, label='compare', s=2)
# plt.legend(loc='upper right')
# plt.title('Absolute errors vs std log probs')


# plt.subplot(3, 2, 2)
# plt.scatter(inv_pred_log_probs, errors, label='inv', s=2)
# plt.scatter(pause_pred_log_probs, errors, label='pause', s=2)
# plt.scatter(compare_pred_log_probs, errors, label='compare', s=2)
# plt.legend(loc='upper right')
# plt.title('Absolute errors vs pred log probs')

# plt.subplot(3, 2, 3)
# plt.scatter(inv_true_y_probs, errors, label='inv', s=2)
# plt.scatter(pause_true_y_probs, errors, label='pause', s=2)
# plt.scatter(compare_true_y_probs, errors, label='compare', s=2)
# plt.legend(loc='upper right')
# plt.title('Absolute errors vs True probs')

# plt.subplot(3, 2, 4)
# plt.scatter(inv_true_y_log_probs, errors, label='inv', s=2)
# plt.scatter(pause_true_y_log_probs, errors, label='pause', s=2)
# plt.scatter(compare_true_y_log_probs, errors, label='compare', s=2)
# plt.legend(loc='upper right')
# plt.title('Absolute errors vs True log probs')

# plt.subplot(3, 2, 5)
# plt.scatter(inv_std, errors, label='inv', s=2)
# plt.scatter(pause_std, errors, label='pause', s=2)
# plt.scatter(compare_std, errors, label='compare', s=2)
# plt.legend(loc='upper right')
# plt.title('Absolute errors vs val Std Dev')

# plt.subplot(3, 2, 6)
# plt.scatter(inv_entropy, errors, label='inv', s=2)
# plt.scatter(pause_entropy, errors, label='pause', s=2)
# plt.scatter(compare_entropy, errors, label='compare', s=2)
# plt.legend(loc='upper right')
# plt.title('Absolute errors vs val Entropy')
# plt.show()


################### DEFERING ###################

# def defer(stddev_inv, stddev_pause, stddev_compare):
# 	stddev_inv = np.argsort(stddev_inv)
# 	stddev_pause = np.argsort(stddev_pause)
# 	stddev_compare = np.argsort(stddev_compare)






# defer_based_on = ensemble_val_stddev
title = 'ensemble_val_stddev'

def defer_value(model_type, true_values, ensemble_preds, defer_based_on):
	true_values = np.squeeze(true_values, axis=-1)
	ensemble_preds = np.squeeze(ensemble_preds, axis=-1)
	defer_based_on = np.squeeze(defer_based_on, axis=-1)
	defered_rmse_list, non_defered_rmse_list = [], []
	for i in range(ensemble_preds.shape[0]+1):
		print('model type {}'.format(model_type))
		print('\n{} datapoints deferred'.format(i))

		if i==ensemble_preds.shape[0]:
			defered_rmse = mean_squared_error(true_values, ensemble_preds, squared=False)
		elif i==0:
			defered_rmse = 0
		else:
			print(true_values[np.argsort(defer_based_on)][-10:].shape)
			defered_rmse = mean_squared_error(
				true_values[np.argsort(defer_based_on)][-i:], 
				ensemble_preds[np.argsort(defer_based_on)][-i:], squared=False)
		defered_rmse_list.append(defered_rmse)

		if i==0:
			non_defered_rmse = mean_squared_error(true_values, ensemble_preds, squared=False)
		elif i==ensemble_preds.shape[0]:
			non_defered_rmse = 0
		else:
			non_defered_rmse = mean_squared_error(
				true_values[np.argsort(defer_based_on)][:-i], 
				ensemble_preds[np.argsort(defer_based_on)][:-i], squared=False)

		non_defered_rmse_list.append(non_defered_rmse)

		
		print('Defered RMSE : {:.3f}'.format(defered_rmse))
		print('Not Defered RMSE : {:.3f}'.format(non_defered_rmse))
	return defered_rmse_list, non_defered_rmse_list

print(ensemble_sigma_inv.shape)
print(ensemble_mu.shape)
print(y_val.shape)

defered_rmse_list, non_defered_rmse_list = defer_value('intervention', y_val, ensemble_mu, ensemble_sigma_inv )

plt.subplot(1, 3, 1)
plt.plot(range(ensemble_mu.shape[0]+1), defered_rmse_list, label='Defered RMSE')
plt.plot(range(ensemble_mu.shape[0]+1), non_defered_rmse_list, label='Non Defered RMSE')
plt.legend()
plt.xlabel('No. of datapoints defered')
plt.xticks(range(ensemble_mu.shape[0]+1))
plt.yticks(range(0,5))
plt.title('Interventions Based on '+title)
plt.grid()

defered_rmse_list, non_defered_rmse_list = defer_value('pause', y_val, ensemble_mu,  ensemble_sigma_pause )

plt.subplot(1, 3, 2)
plt.plot(range(ensemble_mu.shape[0]+1), defered_rmse_list, label='Defered RMSE')
plt.plot(range(ensemble_mu.shape[0]+1), non_defered_rmse_list, label='Non Defered RMSE')
plt.legend()
plt.xlabel('No. of datapoints defered')
plt.xticks(range(ensemble_mu.shape[0]+1))
plt.yticks(range(0,5))
plt.title('Pause Based on '+title)
plt.grid()

defered_rmse_list, non_defered_rmse_list = defer_value('compare', y_val, ensemble_mu, ensemble_sigma_compare )

plt.subplot(1, 3, 3)
plt.plot(range(ensemble_mu.shape[0]+1), defered_rmse_list, label='Defered RMSE')
plt.plot(range(ensemble_mu.shape[0]+1), non_defered_rmse_list, label='Non Defered RMSE')
plt.legend()
plt.xlabel('No. of datapoints defered')
plt.xticks(range(ensemble_mu.shape[0]+1))
plt.yticks(range(0,5))
plt.title('Compare Based on '+title)
plt.grid()

plt.savefig('combined_ensemble_calibration.png')
plt.show()

################### DEFERING ###################



