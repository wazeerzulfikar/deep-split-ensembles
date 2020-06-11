import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions

def create_feature_extractor_block(x, units):
	# x = Dense(16, activation='relu')(x)
	# x = BatchNormalization()(x)
	# x = Dense(8, activation='relu')(x)
	# x = BatchNormalization()(x)
	# x = Dropout(0.2)(x)

	x = Dense(units, activation='relu')(x)
	return x

def create_stddev_block(x):
	# x = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(x)
	x = Dense(1)(x)
	return x

def create_mu_block(x_list):
	x = Concatenate()(x_list)
	# x = Dense(8, activation='relu')(x)
	# x = BatchNormalization()(x)
	# x = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(x)
	x = Dense(1)(x)
	return x

def create_gaussian_output(mu, stddev, name):
	x = Concatenate()([mu, stddev])
	x = tfp.layers.DistributionLambda(
		lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6), name=name)(x)
	return x

def create_multivariate_gaussian_output(mus, stddevs, name):
	x = Concatenate()(mus+stddevs)
	x = tfp.layers.DistributionLambda(
		lambda t: tfd.MultivariateNormalDiag(loc=t[...,:len(mus)], scale_diag=tf.math.softplus(t[...,len(mus):])+1e-6), name=name)(x)
	return x

def build_model(config):
	if config.build_model=='point':
		loss = tf.keras.losses.mean_squared_error
		model = Sequential()
		if config.dataset=='protein' or config.dataset=='msd':
			model.add(Dense(100, activation='relu'))
		else:
			model.add(Dense(50, activation='relu'))
		model.add(Dense(1))

	elif config.build_model=='gaussian':
		loss = lambda y, p_y: -p_y.log_prob(y)
		model = Sequential()
		if config.dataset=='protein' or config.dataset=='msd':
			model.add(Dense(100, activation='relu', dtype='float64'))
		else:
			model.add(Dense(50, activation='relu', dtype='float64') )
		model.add(Dense(2, dtype='float64'))
		model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[..., 1:])+1e-6), dtype='float64'))
		
	elif config.build_model=='combined_pog':
		loss = lambda y, p_y: -p_y.log_prob(y)

		n_feature_sets = len(config.feature_split_lengths)

		inputs = []
		for i in range(n_feature_sets):
			inputs.append(Input((config.feature_split_lengths[i],)))

		feature_extractors = []
		for i in range(n_feature_sets):
			if config.units_type == 'absolute':
				units = config.units
			elif config.units_type == 'prorated':
				units = math.floor(config.feature_split_lengths[i] * config.units / sum(config.feature_split_lengths) )
			feature_extractors.append(create_feature_extractor_block(inputs[i], units = units))

		stddevs = []
		for i in range(n_feature_sets):
			stddevs.append(create_stddev_block(feature_extractors[i]))

		mu = create_mu_block(feature_extractors)

		outputs = []
		for i in range(n_feature_sets):
			outputs.append(create_gaussian_output(mu, stddevs[i], name='set_{}'.format(i)))

		model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

	elif config.build_model=='combined_multivariate':
		loss = lambda y, p_y: -p_y.log_prob(y)

		n_feature_sets = len(config.feature_split_lengths)

		inputs = []
		for i in range(n_feature_sets):
			inputs.append(Input((config.feature_split_lengths[i],)))

		feature_extractors = []
		for i in range(n_feature_sets):
			if config.units_type == 'absolute':
				units = config.units
			elif config.units_type == 'prorated':
				units = math.floor(config.feature_split_lengths[i] * config.units / sum(config.feature_split_lengths) )
			feature_extractors.append(create_feature_extractor_block(inputs[i], units = units))

		stddevs = []
		for i in range(n_feature_sets):
			stddevs.append(create_stddev_block(feature_extractors[i]))

		mu = create_mu_block(feature_extractors)
		mus = [mu]*n_feature_sets

		outputs = create_multivariate_gaussian_output(mus, stddevs, name='mv')

		model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

	elif 'alzheimers' in config.dataset:
		model = alzheimers_model()
		loss = lambda y, p_y: -p_y.log_prob(y)

	else:
		raise Exception('{} model type not available'.format(config.build_model))

	return model, loss

def alzheimers_model():
	intervention_inputs = Input((32, 3))
	pause_inputs = Input((11,))
	compare_inputs = Input((21,))

	intervention_x = LSTM(16)(intervention_inputs)
	intervention_x = BatchNormalization()(intervention_x)

	intervention_x = Dense(16, activation='relu')(intervention_x)
	intervention_x = BatchNormalization()(intervention_x)
	intervention_x = Dense(8, activation='relu')(intervention_x)
	intervention_x = BatchNormalization()(intervention_x)
	# intervention_x = Dropout(0.2)(intervention_x)

	intervention_std = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(intervention_x)

	pause_x = Dense(24, activation='relu')(pause_inputs)
	pause_x = BatchNormalization()(pause_x)
	pause_x = Dense(16, activation='relu')(pause_x)
	pause_x = BatchNormalization()(pause_x)
	pause_x = Dense(16, activation='relu')(pause_x)
	pause_x = BatchNormalization()(pause_x)
	pause_x = Dense(8, activation='relu')(pause_x)
	pause_x = BatchNormalization()(pause_x)
	# pause_x = Dropout(0.2)(pause_x)

	pause_std = Dense(1,  kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(pause_x)

	compare_x = Dense(24, activation='relu')(compare_inputs)
	compare_x = BatchNormalization()(compare_x)
	compare_x = Dense(16, activation='relu')(compare_x)
	compare_x = BatchNormalization()(compare_x)
	compare_x = Dense(16, activation='relu')(compare_x)
	compare_x = BatchNormalization()(compare_x)
	compare_x = Dense(8, activation='relu')(compare_x)
	compare_x = BatchNormalization()(compare_x)
	# compare_x = Dropout(0.2)(compare_x)

	compare_std = Dense(1,  kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(compare_x)

	mu = Concatenate()([intervention_x, pause_x, compare_x])
	# mu = Dense(8, activation='relu')(mu)
	# mu = BatchNormalization()(mu)
	mu = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(mu)

	intervention_gaus = Concatenate()([mu, intervention_std])
	pause_gaus = Concatenate()([mu, pause_std])
	compare_gaus = Concatenate()([mu, compare_std])

	intervention_output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6), name='intervention')(intervention_gaus)
	pause_output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6), name='pause')(pause_gaus)
	compare_output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6), name='compare')(compare_gaus)

	return tf.keras.models.Model(inputs=[intervention_inputs, pause_inputs, compare_inputs],
	 outputs=[intervention_output, pause_output, compare_output])
