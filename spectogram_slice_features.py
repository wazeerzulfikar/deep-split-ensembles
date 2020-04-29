'''
Convolutional and recurrent models for AD classification.
'''


import tensorflow as tf
from tensorflow.keras import layers, regularizers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, TensorBoard
# from keras_self_attention import SeqSelfAttention
# from attention_keras.layers.attention import AttentionLayer

import sys
# sys.path.insert(0, '../input/attention')
# from seq_self_attention import SeqSelfAttention

import os
import glob
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

import numpy as np
np.random.seed(0)

import cv2
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

import spectogram_augmentation
import dataset_features

# spectogram_size = (480, 640)
# spectogram_size = (240, 320)
timesteps_per_slice = 1024
n_mels = 128

dataset_dir = ''
dataset_dir = '../ADReSS-IS2020-data/train/spectograms_np/'
cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'cc/*.npy')))
y_cc = np.zeros((len(cc_files), 2))
y_cc[:,0] = 1

cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'cd/*.npy')))
y_cd = np.zeros((len(cd_files), 2))
y_cd[:,1] = 1


y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
filenames = np.concatenate((cc_files, cd_files), axis=0)

p = np.random.permutation(len(filenames))
y = y[p]
filenames = filenames[p]


# inp_shape = (n_mels, timesteps_per_slice, 1)
inp_shape = (n_mels, None, 1)
print('#####################')
print(inp_shape) # (480, 640, 3)
print('#####################')
num_classes = 2

class DataGenerator():

	def __init__(self, filenames, y, batch_size, split='train'):

		self.filenames = filenames
		# self.X_interventions = X_interventions
		self.y = y
		self.batch_size = batch_size
		self.n_samples = self.filenames.shape[0]
		self.timesteps_per_slice = 1024


	def get_n_batches(self):
		return self.n_samples//self.batch_size


	def flow(self):

		p = np.random.permutation(len(self.filenames))
		self.filenames = self.filenames[p]
		self.y = self.y[p]
		batch_n = 0
		for batch_id in range(self.get_n_batches()):
			batch_filenames = self.filenames[batch_n:batch_n+batch_size]
			y_batch = self.y[batch_n:batch_n+batch_size]

			x_train = [dataset_features.get_sliced_spectogram_features(f, timesteps_per_slice=self.timesteps_per_slice)
			 for f in batch_filenames]
			y_batch_new = []
			x_batch = []
			for e,s in enumerate(x_train):
				label = y_batch[e]
				for i in range(s.shape[0]):
					y_batch_new.append(label)
				if e == 0:
					x_batch = np.reshape(s, (s.shape[0], self.timesteps_per_slice//128, 128, 128, 1))
				else:
					x_batch = np.concatenate((x_batch, 
					 np.reshape(s, (s.shape[0], self.timesteps_per_slice//128, 128, 128, 1))), axis=0)
			y_batch = np.array(y_batch_new)

			batch_n += self.batch_size

			yield (x_batch, y_batch)
			# for x_, y_ in zip(x_batch, y_batch):
			# 	yield (np.expand_dims(x_, axis=0), np.expand_dims(y_, axis=0))

	def on_epoch_end():
		print('Epoch Done!')


def create_model(_type_ = 'convolutional'):

	if _type_=='fully_convolutional':

		model = tf.keras.Sequential()
		model.add(layers.Input(inp_shape))
		# model.add(layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
		#                activation='relu'))
		model.add(layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 2),
						 activation='relu'))
		model.add(layers.MaxPooling2D(pool_size=(2, 2)))
		model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
		#                activation='relu'))
		model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 2),
						 activation='relu'))
		model.add(layers.MaxPooling2D(pool_size=(2, 2)))
		model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
		#                activation='relu'))
		model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
						 activation='relu'))
		model.add(layers.MaxPooling2D(pool_size=(2, 2)))
		model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
		#                activation='relu'))
		model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
						 activation='relu'))
		model.add(layers.MaxPooling2D(pool_size=(2, 2)))
		model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
		#                activation='relu'))
		model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
						 activation='relu'))
		model.add(layers.MaxPooling2D(pool_size=(2, 2)))
		model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1),
		#                activation='relu'))
		# model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2),
		#                activation='relu'))
		# model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2),
		#                activation='relu'))
		# model.add(layers.BatchNormalization())

		# model.add(layers.Flatten())
		# model.add(layers.Dropout(0.5)) # 0.5
		# model.add(layers.GlobalAveragePooling2D())
		# model.add(layers.Dropout(0.5)) # 0.5
		# model.add(layers.Dense(128, activation='relu'))
		# model.add(layers.Dropout(0.5))
		# model.add(layers.Dense(num_classes, activation='softmax'))
		model.add(layers.GlobalAveragePooling2D())
		model.add(layers.Dropout(0.2))
		model.add(layers.Dense(num_classes, activation='softmax'))
		# model.add(layers.Activation('softmax'))

	if _type_ == 'cnn_lstm':
		model = tf.keras.Sequential()
		# model.add(layers.Input(inp_shape))

		model.add(layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 2),
						 activation='relu'))
		model.add(layers.MaxPooling2D(pool_size=(2, 2)))
		model.add(layers.BatchNormalization())


		# model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 2),
		# 				 activation='relu'))
		# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
		# model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
		# 				 activation='relu'))
		# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
		# model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
		# 				 activation='relu'))
		# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
		# model.add(layers.BatchNormalization())

		model.add(layers.Flatten())

		# model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
		# 				 activation='relu'))
		# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
		# model.add(layers.BatchNormalization())

		# model.add(layers.GlobalAveragePooling2D())
		# model.add(layers.Dropout(0.2))
		# model.add(layers.Dense(num_classes, activation='softmax'))
		model_f = tf.keras.Sequential()

		model_f.add(layers.TimeDistributed(layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
						 activation='relu'), input_shape=(None, n_mels, 128, 1)))
		model_f.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
		model_f.add(layers.TimeDistributed(layers.BatchNormalization()))

		model_f.add(layers.TimeDistributed(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
						 activation='relu')))
		model_f.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
		model_f.add(layers.TimeDistributed(layers.BatchNormalization()))

		model_f.add(layers.TimeDistributed(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
						 activation='relu')))
		model_f.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
		model_f.add(layers.TimeDistributed(layers.BatchNormalization()))

		model_f.add(layers.TimeDistributed(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
						 activation='relu')))
		model_f.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
		model_f.add(layers.TimeDistributed(layers.BatchNormalization()))

		model_f.add(layers.TimeDistributed(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
						 activation='relu')))
		model_f.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
		model_f.add(layers.TimeDistributed(layers.BatchNormalization()))

		model_f.add(layers.TimeDistributed(layers.Lambda(lambda x: tf.reduce_max(x, axis=[1,2]))))
		model_f.add(layers.Bidirectional(layers.LSTM(32, activation='sigmoid')))
		model_f.add(layers.BatchNormalization())
		# model.add(layers.Dense(16, activation='relu'))
		# model_f.add(layers.Dropout(0.2))
		model_f.add(layers.Dense(num_classes, activation='softmax'))

		return model_f

	if _type_=='convolutional_1':

		# input_shape_ = (480, 640, 3)
		model2_input = layers.Input(shape=inp_shape,  name='spectrogram_input')
		model2_BN = layers.BatchNormalization()(model2_input)
		
		model2_hidden1 = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
							 activation='relu')(model2_BN)
		# model2_hidden2 = layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2),
		#                    activation='relu')(model2_hidden1)
		model2_BN1 = layers.BatchNormalization()(model2_hidden1)
		model2_hidden2 = layers.MaxPool2D()(model2_BN1)
		
		model2_hidden3 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
							 activation='relu')(model2_hidden2)
		# model2_hidden4 = layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
		#                    activation='relu')(model2_hidden3)
		model2_BN2 = layers.BatchNormalization()(model2_hidden3)
		model2_hidden4 = layers.MaxPool2D()(model2_BN2)

		model2_hidden5 = layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
							 activation='relu')(model2_hidden4)
		# model2_hidden6 = layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
		#                    activation='relu')(model2_hidden5)
		model2_BN3 = layers.BatchNormalization()(model2_hidden5)
		model2_hidden6 = layers.MaxPool2D()(model2_BN3)

		model2_hidden7 = layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1),
							 activation='relu')(model2_hidden6)
		# model2_hidden8 = layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2),
		#                    activation='relu')(model2_hidden7)
		model2_BN4 = layers.BatchNormalization()(model2_hidden7)
		model2_hidden8 = layers.MaxPool2D()(model2_BN4)

		model2_hidden9 = layers.Flatten()(model2_hidden8)
		# model2_hidden10 = layers.Dropout(0.2)(model2_hidden9)
		model2_hidden10 = layers.BatchNormalization()(model2_hidden9)
		model2_hidden11 = layers.Dense(128, activation='relu')(model2_hidden10)
		model2_hidden11 = layers.Dropout(0.2)(model2_hidden11)

		output = layers.Dense(num_classes, activation='softmax')(model2_hidden11)

		return tf.keras.models.Model(inputs=model2_input,outputs=output)

	return model


model = create_model(_type_='cnn_lstm')
# model.build((None, 1024//16, 128, None, 1)) 
print(model.summary())
# exit()

################################# CROSS VALIDATED MODEL TRAINING ################################



n_split = 5

epochs = 40
batch_size = 1

model_dir = '5-fold-models-spectrogram-slice'

val_accuracies = []
train_accuracies = []
fold = 0

train_loop = 'custom'
# train_loop = 'builtin'

spectogram_type = 'sliced'

@tf.function(experimental_relax_shapes=True)
def custom_train_step(x_train, y_train):
	with tf.GradientTape() as tape:
		predictions = model(x_train, training=True)
		loss_value = loss(y_train, predictions)
	grads = tape.gradient(loss_value, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
	return loss_value

for train_index, val_index in KFold(n_split).split(filenames):

	y_train, y_val = y[train_index], y[val_index]
	filenames_train, filenames_val = filenames[train_index], filenames[val_index]

	# if spectogram_type == 'sliced':
	# 	x_val = [dataset_features.get_spectogram_features(f) for f in filenames_val]
	# 	x_val_sliced = [dataset_features.get_sliced_spectogram_features(f, timesteps_per_slice=timesteps_per_slice) for f in filenames_val]

	model = create_model(_type_='cnn_lstm')

	# model = tf.keras.models.load_model('spec_slice.h5')

	timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
	log_name = "{}".format(timeString)


	if train_loop == 'builtin':
		datagen = DataGenerator(filenames_train, y_train, batch_size)

		model.compile(loss=tf.keras.losses.categorical_crossentropy,
			  optimizer=tf.keras.optimizers.Adam(lr=1e-3),
			  metrics=['categorical_accuracy'])

		checkpointer = tf.keras.callbacks.ModelCheckpoint(
			os.path.join(model_dir, 'spec_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=False,
			save_weights_only=False, mode='auto', save_freq='epoch'
		)

		model.fit(datagen.flow(),
				  epochs=epochs,
				  verbose=1,
				  callbacks=[checkpointer])
				  # validation_data=(x_val, y_val))

		# model.fit(datagen.flow(),
		# 		  epochs=epochs,
		# 		  steps_per_epoch=datagen.get_n_batches(),
		# 		  verbose=1,
		# 		  callbacks=[checkpointer],
		# 		  validation_data=(x_val, y_val))

	elif train_loop == 'custom':
		datagen = DataGenerator(filenames_train, y_train, batch_size)

		loss = tf.keras.losses.CategoricalCrossentropy()
		optimizer = tf.keras.optimizers.Adam(lr=1e-4)
		# optimizer = tf.keras.optimizers.SGD()

		for epoch in range(epochs):

			print('Train Epoch')
			progbar = tf.keras.utils.Progbar(len(filenames_train), stateful_metrics=['epoch', 'train_loss', 'train_acc'])
			progbar.update(0, [('epoch', epoch)])
			epoch_loss_avg = tf.keras.metrics.Mean()
			epoch_accuracy = tf.keras.metrics.Mean()

			for x_batch, y_batch in datagen.flow():
				# with tf.GradientTape() as tape:
				# 	predictions = model(x_train_)
				# 	loss_value = loss(y_train_, predictions)
				# grads = tape.gradient(loss_value, model.trainable_variables)
				# optimizer.apply_gradients(zip(grads, model.trainable_variables))
				loss_value = custom_train_step(x_batch, y_batch)
				acc = tf.keras.metrics.categorical_accuracy(y_batch, model(x_batch, training=True))
				epoch_loss_avg.update_state(loss_value)  # Add current batch loss
				# print(y_train_.shape)
				# print( model(x_train_).shape)
				epoch_accuracy.update_state(acc)
				progbar.add(batch_size, [('train_loss', epoch_loss_avg.result()),
					('train_acc', epoch_accuracy.result())])

			epoch_loss_avg.reset_states()
			epoch_accuracy.reset_states()

			print()
			print('Val Epoch')
			progbar = tf.keras.utils.Progbar(len(filenames_val), stateful_metrics=['epoch', 'val_loss'])
			progbar.update(0, [('epoch', epoch)])
			epoch_loss_avg = tf.keras.metrics.Mean()
			epoch_accuracy = tf.keras.metrics.Mean()

			for x_val_batch_filename, y_val_batch in zip(filenames_val, y_val):
				x_val_batch = np.array(dataset_features.get_spectogram_features(x_val_batch_filename))
				x_val_batch = x_val_batch[:,:-(x_val_batch.shape[1]%128),:]
				# x_val_batch_ = x_val_batch_[:,:1024,:]
				x_val_batch = np.reshape(x_val_batch, (-1, 128, 128, 1))
				pred = model(np.expand_dims(x_val_batch, axis=0), training=False)
				loss_value = loss(y_val_batch, pred)
				acc = tf.keras.metrics.categorical_accuracy(y_val_batch, pred)
				epoch_loss_avg.update_state(loss_value)  # Add current batch loss
				epoch_accuracy.update_state(acc)
				progbar.add(1, [('val_loss', epoch_loss_avg.result()),
					('val_acc', epoch_accuracy.result())])

			epoch_loss_avg.reset_states()
			epoch_accuracy.reset_states()

			print('Val Epoch first timesteps_per_slice')
			progbar = tf.keras.utils.Progbar(len(filenames_val), stateful_metrics=['epoch', 'val_loss', 'val_acc', 'voting_acc'])
			progbar.update(0, [('epoch', epoch)])
			epoch_loss_avg = tf.keras.metrics.Mean()
			epoch_accuracy = tf.keras.metrics.Mean()
			voting_epoch_accuracy = tf.keras.metrics.Mean()

			for x_val_batch_filename, y_val_batch in zip(filenames_val, y_val):
				x_val_batch = np.array(dataset_features.get_sliced_spectogram_features(x_val_batch_filename, timesteps_per_slice=timesteps_per_slice))
				x_val_batch = np.reshape(x_val_batch, (-1, timesteps_per_slice//128, 128, 128, 1))
				preds = model(x_val_batch, training=False)
				mean_preds = np.mean(preds, axis=0)
				loss_value = loss(y_val_batch, mean_preds)
				acc = tf.keras.metrics.categorical_accuracy(y_val_batch, mean_preds)

				# preds = np.argmax(preds, axis=-1)
				# print(preds)
				# preds_values, counts = np.unique(preds, return_counts=True)
				# print(preds_values, counts)
				# preds = [list(map(lambda x: 0.0 if x <0.5 else 1.0, i)) for i in preds]
				preds = np.argmax(preds, axis=-1)
				preds_values, counts = np.unique(preds, return_counts=True)
				# print(preds_values, counts)
				# print(preds)
				# preds = preds_values[np.argmax(counts)]
				voting_acc = tf.keras.metrics.categorical_accuracy(y_val_batch, preds)
				epoch_loss_avg.update_state(loss_value)  # Add current batch loss
				epoch_accuracy.update_state(acc)
				voting_epoch_accuracy.update_state(voting_acc)
				progbar.add(1, [('val_loss', epoch_loss_avg.result()),
					('val_acc', epoch_accuracy.result()),
					('voting_acc', voting_epoch_accuracy.result())])

			epoch_loss_avg.reset_states()
			epoch_accuracy.reset_states()
			voting_epoch_accuracy.reset_states()
			print()
			model.save('spec_slice.h5')

	model = tf.keras.models.load_model(os.path.join(model_dir, 'spec_{}.h5'.format(fold)))

	val_pred = model.predict(x_val)
	for i in range(len(x_val)):
		print(filenames_val[i], np.argmax(val_pred[i])==np.argmax(y_val[i]), val_pred[i])

	train_score = model.evaluate(x_train, y_train, verbose=0)
	# print('_'*30)
	# print(model.predict(x_train))
	# print(model.predict(filenames_val))

	train_accuracies.append(train_score[1])
	score = model.evaluate(filenames_val, y_val, verbose=0)
	print('Val accuracy:', score[1])
	val_accuracies.append(score[1])

	print('Train accuracies ', train_accuracies)
	print('Train mean', np.mean(train_accuracies))
	print('Train std', np.std(train_accuracies))

	print('Val accuracies ', val_accuracies)
	print('Val mean', np.mean(val_accuracies))
	print('Val std', np.std(val_accuracies))

	fold+=1

