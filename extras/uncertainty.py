import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import *

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import glob, os, math, time
from decimal import *
import numpy as np
np.random.seed(0)
from pathlib import Path
import seaborn as sns

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import dataset

tfd = tfp.distributions

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

def train_a_fold(model_type, x_train, y_train, x_val, y_val, model_dir, model_number, fold=1):

	if model_type == 'pause':
		model = create_pause_model()
		lr = 0.01
		epochs = 5000

	elif model_type == 'intervention':
		model = create_intervention_model()
		lr = 0.001
		epochs = 2000

	elif model_type == 'compare':
		model = create_compare_model()
		lr = 0.001
		epochs = 15000	

	########### COMMENT FOR EVALUATION ###########
	negloglik = lambda y, p_y: -p_y.log_prob(y)

	class MeanMetricWrapper(tf.keras.metrics.Mean):
		def __init__(self, fn, name=None, dtype=None, **kwargs):
			super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
			self._fn = fn
			self._fn_kwargs = kwargs
		def update_state(self, y_true, y_pred, sample_weight=None):
			matches = self._fn(y_true, y_pred, **self._fn_kwargs)
			return super(MeanMetricWrapper, self).update_state(
				matches, sample_weight=sample_weight)
		def get_config(self):
			config = {}
			for k, v in six.iteritems(self._fn_kwargs):
				config[k] = tf.keras.backend.eval(v) if is_tensor_or_variable(v) else v
			base_config = super(MeanMetricWrapper, self).get_config()
			return dict(list(base_config.items()) + list(config.items()))

	custom_mse = lambda y, p_y: tf.keras.losses.mean_squared_error(y, p_y.mean())
	
	checkpoint_filepath = os.path.join(model_dir, model_type, 'model_nll_{}.h5'.format(model_number))
	checkpointer_nll = tf.keras.callbacks.ModelCheckpoint(
			checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
			save_weights_only=True, mode='min', save_freq='epoch')

	checkpoint_filepath = os.path.join(model_dir, model_type, 'model_mse_{}.h5'.format(model_number))
	checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(
			checkpoint_filepath, monitor='val_custom_mse', verbose=0, save_best_only=True,
			save_weights_only=True, mode='min', save_freq='epoch')

	tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs-uncertainty/911-{}".format(model_number), histogram_freq=1, write_graph=True, write_images=False)

	model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
				  loss=negloglik, metrics=[MeanMetricWrapper(custom_mse, name='custom_mse')])

	hist = model.fit(x_train, y_train,
					batch_size=8,
					epochs=epochs,
					verbose=1,
					callbacks=[checkpointer_nll, checkpointer_mse, tensorboard],
					validation_data=(x_val, y_val))

	epoch_val_losses = hist.history['val_custom_mse']
	best_epoch_val_loss, best_epoch = np.min(epoch_val_losses), np.argmin(epoch_val_losses)+1
	best_epoch_train_loss = hist.history['custom_mse'][best_epoch-1]
	########### COMMENT FOR EVALUATION ###########

	checkpoint_filepath = os.path.join(model_dir, model_type, 'model_mse_{}.h5'.format(model_number)) ########### CHANGE WHICH MODEL TO LOAD nll OR mse
	model.load_weights(checkpoint_filepath)

	# return model ########### COMMENT FOR TRAINING
	return model, best_epoch_train_loss, best_epoch_val_loss, best_epoch ########### COMMENT FOR EVALUATION

data = dataset.prepare_data('../ADReSS-IS2020-data/train')
X_intervention, X_pause, X_spec, X_compare, X_reg_intervention, X_reg_pause, X_reg_compare = data[0:7]
y, y_reg, filenames_intervention, filenames_pause, filenames_spec, filenames_compare = data[7:]

feature_types = {
	'intervention': X_reg_intervention,
	'pause': X_reg_pause,
	'compare': X_reg_compare
}

results = {}


model_types = ['pause'] # # 'uncertainty-2' mse + nll saving running in tmux 6
# model_types = ['intervention'] # 'uncertainty-2' mse + nll saving running in tmux 2
model_types = ['compare'] # 'uncertainty-2' mse + nll saving running in tmux 0

model_dir = 'uncertainty'
model_dir = 'uncertainty-2'
model_dir = Path(model_dir)
model_dir.joinpath(model_types[0]).mkdir(parents=True, exist_ok=True)
model_dir = str(model_dir)

model_type = model_types[0]
x, y = feature_types[model_type], y_reg
n_samples = x.shape[0]


n_models = 10
mus, sigmas = [], []
train_nlls, val_nlls = [], []
start = time.time()
for model_number in range(n_models):
	print('~ Training model {} ~'.format(model_number+1))

	x_train, x_val = x[int(0.2*n_samples):], x[:int(0.2*n_samples)]
	y_train, y_val = y[int(0.2*n_samples):], y[:int(0.2*n_samples)]

	if model_type=='compare':
		features_size = 21
		sc = StandardScaler()
		sc.fit(x_train)

		x_train = sc.transform(x_train)
		x_val = sc.transform(x_val)

		pca = PCA(n_components=features_size)
		pca.fit(x_train)

		x_train = pca.transform(x_train)
		x_val = pca.transform(x_val)	

	n_val_samples = x_val.shape[0]

	model, best_epoch_train_loss, best_epoch_val_loss, best_epoch = train_a_fold(model_type, x_train, y_train, x_val, y_val, model_dir, str(model_number+1)) ########### COMMENT FOR EVALUATION
	# model = train_a_fold(model_type, x_train, y_train, x_val, y_val, model_dir, str(model_number+1)) ########### COMMENT FOR TRAINING
	train_nll, val_nll = best_epoch_train_loss, best_epoch_val_loss ########### COMMENT FOR EVALUATION

	y_val = y_val.reshape(-1,1)
	pred = model(x_val)
	
	mu = pred.mean()
	sigma = pred.stddev()

	mus.append(mu.numpy())
	sigmas.append(sigma.numpy())
	train_nlls.append(train_nll) ########### COMMENT FOR EVALUATION
	val_nlls.append(val_nll) ########### COMMENT FOR EVALUATION

	val_rmse = mean_squared_error(y_val, mu, squared=False)

	########### COMMENT FOR EVALUATION ###########
	print()
	print(model_type)
	print('Best Epoch: {:d}'.format(best_epoch))
	print('Train RMSE: {:.3f}'.format(best_epoch_train_loss)) 
	print('Val RMSE: {:.3f}'.format(best_epoch_val_loss)) # NLL
	print('Val RMSE: {:.3f}'.format(val_rmse))

	pred_probs = pred.prob(mu)
	true_y_probs = pred.prob(y_val)
	print()
	for i in range(n_val_samples):
		tf.print('Pred: {:.3f}'.format(mu[i][0]), '\t\t\tProb: {:.7f}'.format(pred_probs[i][0]), '\t\t\tTrue: {}'.format(y_val[i][0]), '\t\t\tProb: {:.7f}'.format(true_y_probs[i][0]), '\t\t\tStd Dev: {:.7f}'.format(sigma[i][0]), '\t\t\tEntropy: {:.7f}'.format(pred.entropy()[i][0]))
	print()
	print(model_type)
	########### COMMENT FOR EVALUATION ###########

mus, sigmas = np.concatenate(mus, axis=-1), np.concatenate(sigmas, axis=-1)
ensemble_mu = np.mean(mus, axis=-1).reshape(-1,1)
ensemble_sigma = np.sqrt(np.mean(np.square(sigmas) + np.square(mus), axis=-1).reshape(-1,1) - np.square(ensemble_mu))

ensemble_dist = tfd.Normal(loc=ensemble_mu, scale=ensemble_sigma)
ensemble_pred_probs = ensemble_dist.prob(ensemble_mu).numpy()
ensemble_pred_log_probs = ensemble_dist.log_prob(ensemble_mu).numpy()
ensemble_true_probs = ensemble_dist.prob(y_val).numpy()
ensemble_true_log_probs = ensemble_dist.log_prob(y_val).numpy()
ensemble_nll = np.mean(-ensemble_true_log_probs)

print()
print('Deep ensemble results')
for i in range(n_val_samples):
	print('Pred: {:.3f}'.format(ensemble_mu[i][0]), '\t\t\tProb: {:.7f}'.format(ensemble_pred_probs[i][0]), '\t\t\tTrue: {}'.format(y_val[i][0]), '\t\t\tProb: {:.7f}'.format(ensemble_true_probs[i][0]), '\t\t\tStd Dev: {:.7f}'.format(ensemble_sigma[i][0]), '\t\t\tEntropy: {:.7f}'.format(ensemble_dist.entropy()[i][0]))
print()
########### COMMENT FOR EVALUATION ###########
print('Train NLLs: ', train_nlls)
print('Val NLLs: ',val_nlls)
print('Train NLL: {:.3f} +/- {:.3f}'.format(np.mean(train_nlls), np.std(train_nlls)))
print('Val NLL: {:.3f} +/- {:.3f}'.format(np.mean(val_nlls), np.std(val_nlls)))
########### COMMENT FOR EVALUATION ###########
print('Ensemble Val NLL calculated using ensemble distribution: {:.3f}'.format(ensemble_nll))
print('Deep ensemble of {} models took {} minutes'.format(n_models, int((time.time()-start)/60)))

min_prob = float(Decimal(str(np.min(ensemble_pred_probs))).quantize(Decimal('.001'), rounding=ROUND_DOWN))
max_prob = float(Decimal(str(np.max(ensemble_pred_probs))).quantize(Decimal('.001'), rounding=ROUND_DOWN))
bins = np.linspace(min_prob, max_prob, num=10)

binned_rmses = []
for b in bins:
	conditioned_preds = ensemble_mu[ensemble_pred_probs>b]
	conditioned_true = y_val[ensemble_pred_probs>b]
	bin_rmse = mean_squared_error(conditioned_true, conditioned_preds, squared=False)
	binned_rmses.append(bin_rmse)

ensemble_val_rmse = mean_squared_error(y_val, ensemble_mu, squared=False)

print('Bins: ', list(bins))
print('Binned RMSEs: ', list(binned_rmses))
print()
print('Ensemble Preds: ', list(np.squeeze(ensemble_mu)))
print('True Values: ', list(np.squeeze(y_val)))
print('Ensemble Pred Probs: ', list(np.squeeze(ensemble_pred_probs)))
print('Ensemble Pred Log Probs: ', list(np.squeeze(ensemble_pred_log_probs)))
print('Ensemble True Probs: ', list(np.squeeze(ensemble_true_probs)))
print('Ensemble True Log Probs: ', list(np.squeeze(ensemble_true_log_probs)))
print('Ensemble val stddev: ', list(np.squeeze(ensemble_dist.stddev())))
print('Ensemble Val Entropy: ', list(np.squeeze(ensemble_dist.entropy())))
print('Ensemble Val RMSE: ', ensemble_val_rmse)

'''
########### INTERVENTIONS ##############
'''
'''
Upon saving based on NLL

Ensemble Preds:  [19.76003, 19.571033, 12.125517, 23.078472, 19.34019, 14.236501, 19.651865, 19.705332, 18.831812, 18.828154, 19.721909, 19.701374, 19.729279, 19.560772, 19.917545, 19.68107, 19.563038, 21.08012, 14.409856, 10.760031, 20.14383]
True Values:  [13.0, 30.0, 13.0, 29.0, 28.0, 14.0, 13.0, 30.0, 27.0, 16.0, 17.0, 29.0, 29.0, 18.0, 16.0, 28.0, 28.0, 27.0, 19.0, 11.0, 20.0]
Ensemble Pred Probs:  [0.066326015, 0.065911815, 0.06148366, 0.042929165, 0.065179266, 0.060637724, 0.065800585, 0.065442175, 0.061963398, 0.06668946, 0.066085055, 0.06551437, 0.06595322, 0.06576934, 0.06449749, 0.065160036, 0.065194026, 0.0670268, 0.063569225, 0.05844211, 0.06631085]
Ensemble Pred Log Probs:  [-2.7131732, -2.7194376, -2.7889838, -3.1482038, -2.730614, -2.802838, -2.7211266, -2.7265882, -2.7812114, -2.7077084, -2.7168126, -2.7254858, -2.7188096, -2.7216015, -2.741129, -2.7309089, -2.7303874, -2.7026627, -2.7556257, -2.8397186, -2.7134018]
Ensemble True Probs:  [0.035269603, 0.014937744, 0.0609278, 0.035041716, 0.023956835, 0.060598556, 0.036044985, 0.015725188, 0.027709449, 0.059637886, 0.059697706, 0.020417081, 0.020378025, 0.06362778, 0.052776035, 0.025887204, 0.025201399, 0.040873066, 0.048649736, 0.058406003, 0.0662919]
Ensemble True Log Probs:  [-3.3447337, -4.203864, -2.7980657, -3.351216, -3.7315016, -2.8034842, -3.3229876, -4.1524916, -3.5859818, -2.8194642, -2.8184617, -3.8913834, -3.8932981, -2.7547052, -2.941698, -3.6540065, -3.6808558, -3.197284, -3.023109, -2.8403366, -2.7136877]
Ensemble val stddev:  [6.014869, 6.0526686, 6.4885907, 9.293036, 6.1206937, 6.5791106, 6.0628977, 6.096103, 6.438353, 5.98209, 6.036801, 6.0893865, 6.048868, 6.0657787, 6.185391, 6.1224985, 6.1193075, 5.9519815, 6.2757125, 6.826281, 6.016244]
Ensemble Val RMSE:  6.489795
'''

'''
Upon saving based on MSE trial corresponding

Train NLLs:  [56.66371536254883, 42.180908203125, 51.627349853515625, 76.04356384277344, 56.3496208190918, 58.90558624267578, 59.0626335144043, 54.169830322265625, 55.3079719543457, 59.29501724243164]
Val NLLs:  [37.24531555175781, 44.704978942871094, 35.46138381958008, 39.861915588378906, 45.06169891357422, 36.081787109375, 40.343406677246094, 38.255821228027344, 37.22850799560547, 39.682861328125]
Train NLL: 56.961 +/- 7.973
Val NLL: 39.393 +/- 3.140
Ensemble Val NLL calculated using ensemble distribution: 3.229
Deep ensemble of 10 models took 68 minutes
Bins:  [0.049, 0.051555555555555556, 0.05411111111111111, 0.056666666666666664, 0.05922222222222222, 0.06177777777777778, 0.06433333333333333, 0.06688888888888889, 0.06944444444444445, 0.072]
Binned RMSEs:  [5.9304457, 5.821265, 5.821265, 5.821265, 6.242784, 6.242784, 6.434245, 6.823451, 6.6704717, 9.592852]

Ensemble Preds:  [22.592852, 21.448545, 13.755903, 21.200565, 21.818558, 16.980942, 21.100103, 21.858631, 21.338009, 20.654617, 22.405613, 22.03941, 22.699509, 20.763569, 22.42091, 21.899014, 20.974804, 23.174145, 18.628778, 13.399005, 22.443352]
True Values:  [13.0, 30.0, 13.0, 29.0, 28.0, 14.0, 13.0, 30.0, 27.0, 16.0, 17.0, 29.0, 29.0, 18.0, 16.0, 28.0, 28.0, 27.0, 19.0, 11.0, 20.0]
Ensemble Pred Probs:  [0.072088696, 0.06901411, 0.057687514, 0.049116496, 0.07044083, 0.058882434, 0.06556543, 0.06890955, 0.06741659, 0.06461476, 0.0683273, 0.07146095, 0.071731985, 0.064823486, 0.07066407, 0.07044751, 0.066841185, 0.070181645, 0.061926093, 0.058086783, 0.06581957]
Ensemble Pred Log Probs:  [-2.629858, -2.6734443, -2.8527145, -3.0135603, -2.6529822, -2.8322124, -2.7247066, -2.6749606, -2.6968641, -2.7393124, -2.683446, -2.6386042, -2.6348186, -2.7360873, -2.649818, -2.6528873, -2.7054358, -2.6566684, -2.7818136, -2.845817, -2.720838]
Ensemble True Probs:  [0.016046936, 0.02310591, 0.057343923, 0.03097447, 0.038827963, 0.05345036, 0.027030142, 0.025636235, 0.042655136, 0.048631467, 0.044510845, 0.032847222, 0.037759904, 0.05860652, 0.037009742, 0.0394301, 0.033435013, 0.055957597, 0.061823376, 0.054649115, 0.06068312]
Ensemble True Log Probs:  [-4.1322374, -3.7676668, -2.8586884, -3.474592, -3.2486145, -2.9290018, -3.6108027, -3.6637485, -3.1546075, -3.0234845, -3.1120224, -3.415888, -3.2765074, -2.8369093, -3.296574, -3.2332258, -3.3981516, -2.883161, -2.7834737, -2.9068222, -2.8020897]
Ensemble val stddev:  [5.5340466, 5.78059, 6.915573, 8.122369, 5.6635103, 6.775235, 6.084644, 5.789363, 5.9175677, 6.174166, 5.838695, 5.582661, 5.5615673, 6.154287, 5.6456165, 5.6629715, 5.9685097, 5.6844244, 6.4422317, 6.8680387, 6.0611506]
Ensemble Val RMSE:  5.9304457
'''

'''
Upon saving based on NLL trial corresponding

Ensemble Val NLL calculated using ensemble distribution: 3.299
Deep ensemble of 10 models took 0 minutes
Bins:  [0.05, 0.052111111111111115, 0.05422222222222223, 0.05633333333333334, 0.05844444444444445, 0.06055555555555556, 0.06266666666666668, 0.06477777777777778, 0.06688888888888889, 0.069]
Binned RMSEs:  [6.4811916, 6.422876, 6.422876, 6.422876, 6.422876, 6.5883126, 6.522581, 6.705185, 6.8590946, 6.497358]

Ensemble Preds:  [20.616007, 19.600424, 10.735434, 21.446451, 19.725674, 14.596051, 19.96127, 20.137886, 19.324179, 19.700907, 20.536327, 19.73283, 20.796623, 19.806614, 20.817057, 19.880224, 19.711012, 21.82233, 15.805769, 9.782767, 21.161045]
True Values:  [13.0, 30.0, 13.0, 29.0, 28.0, 14.0, 13.0, 30.0, 27.0, 16.0, 17.0, 29.0, 29.0, 18.0, 16.0, 28.0, 28.0, 27.0, 19.0, 11.0, 20.0]
Ensemble Pred Probs:  [0.06906843, 0.0659168, 0.065581284, 0.050202683, 0.067133784, 0.060209822, 0.067367435, 0.06739496, 0.06124644, 0.067405015, 0.06526125, 0.06686394, 0.069153674, 0.06641848, 0.066905, 0.06775928, 0.06610119, 0.06889305, 0.06594086, 0.06415004, 0.06942292]
Ensemble Pred Log Probs:  [-2.6726575, -2.719362, -2.724465, -2.9916868, -2.701068, -2.8099198, -2.6975935, -2.697185, -2.7928495, -2.6970358, -2.7293568, -2.7050955, -2.6714242, -2.71178, -2.7044816, -2.691794, -2.7165685, -2.6752, -2.718997, -2.7465305, -2.6675382]
Ensemble True Probs:  [0.028956924, 0.015060833, 0.061190933, 0.03195443, 0.025464876, 0.05996669, 0.033758868, 0.016822433, 0.03058724, 0.05543536, 0.055205997, 0.020014066, 0.025161054, 0.06348108, 0.048277386, 0.026179379, 0.025740284, 0.046192765, 0.057361946, 0.06293291, 0.06802033]
Ensemble True Log Probs:  [-3.541946, -4.1956577, -2.7937562, -3.4434445, -3.6704552, -2.813966, -3.3885121, -4.085042, -3.4871724, -2.8925376, -2.8966837, -3.91132, -3.682458, -2.7570133, -3.030792, -3.6427832, -3.659698, -3.074932, -2.858374, -2.765686, -2.6879487]
Ensemble val stddev:  [5.776043, 6.05221, 6.0831733, 7.946634, 5.9424963, 6.6258683, 5.9218855, 5.919468, 6.513721, 5.918584, 6.113003, 5.9664793, 5.768924, 6.006497, 5.962816, 5.88764, 6.035327, 5.790749, 6.0500007, 6.218893, 5.746549]
Ensemble Val Entropy:  [3.1726575, 3.219362, 3.224465, 3.4916868, 3.201068, 3.3099198, 3.1975935, 3.197185, 3.2928495, 3.1970358, 3.2293568, 3.2050955, 3.1714242, 3.21178, 3.2044816, 3.191794, 3.2165685, 3.1752, 3.218997, 3.2465305, 3.1675382]
Ensemble Val RMSE:  6.4811916
'''

'''
########### PAUSE ##############
'''
'''
Upon saving based on MSE trial corresponding

Ensemble Val NLL calculated using ensemble distribution: 3.232
Deep ensemble of 10 models took 0 minutes
Bins:  [0.056, 0.059111111111111114, 0.06222222222222222, 0.06533333333333334, 0.06844444444444445, 0.07155555555555555, 0.07466666666666667, 0.07777777777777778, 0.0808888888888889, 0.084]
Binned RMSEs:  [6.1987734, 5.9050875, 5.9413548, 5.8340225, 5.70962, 5.288861, 5.0907497, 5.325179, 5.1710806, 3.57263]

Ensemble Preds:  [21.616798, 24.174332, 18.98918, 21.84357, 20.832342, 20.858896, 20.450039, 24.057964, 22.62692, 20.135294, 20.676428, 22.87759, 23.930386, 16.87083, 16.557745, 24.42737, 15.996732, 19.67284, 23.995039, 18.820564, 23.356327]
True Values:  [13.0, 30.0, 13.0, 29.0, 28.0, 14.0, 13.0, 30.0, 27.0, 16.0, 17.0, 29.0, 29.0, 18.0, 16.0, 28.0, 28.0, 27.0, 19.0, 11.0, 20.0]
Ensemble Pred Probs:  [0.07034701, 0.08211146, 0.0631613, 0.07270612, 0.06814593, 0.06623952, 0.0658004, 0.0812626, 0.073609255, 0.06539595, 0.066728935, 0.07789926, 0.08046252, 0.05912496, 0.05737434, 0.08405996, 0.056407627, 0.06403576, 0.08137893, 0.062155575, 0.076432906]
Ensemble Pred Log Probs:  [-2.654315, -2.4996777, -2.7620635, -2.6213298, -2.6861038, -2.714478, -2.7211294, -2.5100694, -2.6089845, -2.727295, -2.7071166, -2.5523388, -2.5199637, -2.828102, -2.858158, -2.476225, -2.875151, -2.7483137, -2.5086389, -2.7781148, -2.571342]
Ensemble True Probs:  [0.022178035, 0.04001324, 0.04029119, 0.031059535, 0.03220525, 0.034632865, 0.030928105, 0.039063774, 0.053156585, 0.051972028, 0.055233277, 0.038123064, 0.04770567, 0.058302827, 0.057190064, 0.06331945, 0.013361788, 0.03206687, 0.048424967, 0.02958682, 0.062157188]
Ensemble True Log Probs:  [-3.8086529, -3.218545, -3.2116225, -3.4718494, -3.4356258, -3.3629522, -3.47609, -3.2425597, -2.9345133, -2.9570496, -2.8961897, -3.2669358, -3.042705, -2.8421047, -2.861375, -2.7595627, -4.3153563, -3.4399319, -3.0277398, -3.5204263, -2.7780888]
Ensemble val stddev:  [5.6710625, 4.858546, 6.3162456, 5.4870534, 5.8542347, 6.022723, 6.0629153, 4.909297, 5.41973, 6.100412, 5.978551, 5.121259, 4.958113, 6.7474427, 6.9533224, 4.745925, 7.072489, 6.2299924, 4.90228, 6.418448, 5.21951]
Ensemble Val Entropy:  [3.154315, 2.9996777, 3.2620635, 3.1213298, 3.1861038, 3.214478, 3.2211294, 3.0100694, 3.1089845, 3.227295, 3.2071166, 3.0523388, 3.0199637, 3.328102, 3.358158, 2.976225, 3.375151, 3.2483137, 3.0086389, 3.2781148, 3.071342]
Ensemble Val RMSE:  6.1987734
'''

'''
Upon saving based on NLL trial corresponding

Ensemble Val NLL calculated using ensemble distribution: 3.301
Deep ensemble of 10 models took 0 minutes
Bins:  [0.05, 0.05255555555555556, 0.05511111111111111, 0.057666666666666665, 0.06022222222222222, 0.06277777777777778, 0.06533333333333333, 0.06788888888888889, 0.07044444444444445, 0.073]
Binned RMSEs:  [6.7332144, 6.118215, 6.2668867, 6.2668867, 6.4190245, 5.894279, 5.3628435, 5.2966204, 5.2966204, 5.845009]

Ensemble Preds:  [18.871563, 22.313894, 17.594662, 19.27821, 18.941639, 17.775421, 16.955332, 23.247786, 19.615765, 17.452942, 20.282528, 17.729937, 23.173899, 15.870738, 14.435493, 23.208704, 13.7378645, 18.334694, 22.118391, 17.127201, 21.125875]
True Values:  [13.0, 30.0, 13.0, 29.0, 28.0, 14.0, 13.0, 30.0, 27.0, 16.0, 17.0, 29.0, 29.0, 18.0, 16.0, 28.0, 28.0, 27.0, 19.0, 11.0, 20.0]
Ensemble Pred Probs:  [0.06288631, 0.06786773, 0.061907552, 0.062092416, 0.061777655, 0.06022646, 0.0617975, 0.07368669, 0.06282872, 0.06150272, 0.06462115, 0.06103071, 0.07354723, 0.0596322, 0.054885406, 0.07384463, 0.050932363, 0.063670434, 0.07125341, 0.06237674, 0.065827325]
Ensemble Pred Log Probs:  [-2.7664268, -2.6901946, -2.782113, -2.7791314, -2.7842135, -2.8096435, -2.7838924, -2.607933, -2.767343, -2.7886739, -2.7392135, -2.7963781, -2.6098275, -2.8195596, -2.9025078, -2.605792, -2.9772568, -2.754035, -2.6415126, -2.7745628, -2.7207203]
Ensemble True Probs:  [0.040976852, 0.02886738, 0.04801246, 0.019763768, 0.023097632, 0.051197246, 0.051221825, 0.033855688, 0.031951174, 0.059979044, 0.05610278, 0.013805975, 0.041310225, 0.056687105, 0.05362864, 0.04983367, 0.009706503, 0.024469377, 0.061016172, 0.039420698, 0.06470115]
Ensemble True Log Probs:  [-3.194748, -3.545043, -3.0362947, -3.923905, -3.7680252, -2.9720695, -2.9715896, -3.3856483, -3.4435463, -2.81376, -2.88057, -4.282654, -3.1866453, -2.8702085, -2.925672, -2.9990644, -4.634959, -3.7103329, -2.7966163, -3.2334642, -2.7379763]
Ensemble val stddev:  [6.3438654, 5.8782325, 6.444162, 6.424975, 6.4577107, 6.6240373, 6.455638, 5.4140344, 6.3496814, 6.4865794, 6.173556, 6.5367475, 5.4243, 6.690049, 7.268641, 5.4024553, 7.832786, 6.2657394, 5.5989223, 6.3956895, 6.060436]
Ensemble Val Entropy:  [3.2664268, 3.1901946, 3.282113, 3.2791314, 3.2842135, 3.3096435, 3.2838924, 3.107933, 3.267343, 3.2886739, 3.2392135, 3.2963781, 3.1098275, 3.3195596, 3.4025078, 3.105792, 3.4772568, 3.254035, 3.1415126, 3.2745628, 3.2207203]
Ensemble Val RMSE:  6.7332144
'''
