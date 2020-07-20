import os
import numpy as np
from scipy import misc
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops

def scipy_resize(img):
    img = misc.imresize(img, size=(8, 8), interp='bicubic').astype(np.float32) / 255.
    return img


def scipy_resize_nearest(img, size):
    img = misc.imresize(img, size=(size, size), interp='nearest').astype(np.float32) / 255.
    return img


def zeroCenter(I):
    return 2 * I - 1


def revertZeroCenter(I):
    return I / 2 + 0.5


def get_images(latent_z, model, batch_size):
    latent = tf.concat([latent_z[i] for i in range(batch_size)], axis=0)
    w_tmp = model.components.mapping.get_output_for(latent, None, dlatent_broadcast=None)
    w = tf.concat([tf.expand_dims(l, axis=0) for l in
                   tf.split(w_tmp,
                            batch_size, axis=0)], axis=0)

    images_chw = model.components.synthesis.get_output_for(w,
                                                           randomize_noise=False,
                                                           is_training=False,
                                                           use_noise=False)

    images = tf.transpose(images_chw, perm=[0, 2, 3, 1])

    return images

def create_dir(new_dir):
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)

def load_test_sample(im_path, a_path):
	tf.executing_eagerly()

	# Preprocessing High-Resolution Input Image
	raw_image = tf.read_file(im_path)
	hr_input_image = tf.image.convert_image_dtype(tf.image.decode_png(raw_image, channels=3), dtype=tf.float32)
	hr_input_image.set_shape([128, 128, 3])

	# Generating Low-Resolution Image By Downsampling High-Resolution Image 16 Times
	low_res_image = tf.py_func(scipy_resize, [hr_input_image], tf.float32)
	low_res_image_rep = tf.py_func(scipy_resize_nearest, [low_res_image, 128], tf.float32)

	# Preprocessing Audio Input
	raw_audio = tf.read_file(a_path)
	decoded_audio = audio_ops.decode_wav(raw_audio, desired_channels=1).audio
	decoded_audio.set_shape([64000, 1])
	spectrogram = audio_ops.audio_spectrogram(decoded_audio,
	                                          window_size=512,
	                                          stride=248,
	                                          magnitude_squared=True)

	logSpectrogram = tf.expand_dims(tf.squeeze(tf.log(spectrogram + 1e-6), axis=0), axis=2)

	min_val = tf.reduce_min(logSpectrogram, axis=[0, 1],
	                        keepdims=True)

	max_val = tf.reduce_max(logSpectrogram, axis=[0, 1], keepdims=True)

	logSpectrogram = tf.divide(logSpectrogram + tf.abs(min_val), max_val - min_val)

	# Adding Batch Dimenstion and Adjusting Input Ranges
	hr_input_image = tf.expand_dims(hr_input_image, axis=0)
	low_res_image = tf.expand_dims(zeroCenter(low_res_image), axis=0)
	logSpectrogram  = tf.expand_dims(zeroCenter(logSpectrogram),axis=0)
	low_res_image_rep = tf.expand_dims(low_res_image_rep, axis=0)

	return hr_input_image, low_res_image, logSpectrogram, low_res_image_rep