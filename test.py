import os
import imageio
import tensorflow as tf
import numpy as np
from networks import LowResEncoder, SpectrogramEncoder, FC3ResidualFuser
import pickle
import dnnlib.tflib as tflib

from utils import zeroCenter, revertZeroCenter, get_images, create_dir, load_test_sample

import glob

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('BATCH_SIZE', 1, '')
flags.DEFINE_string('DATA_PATH', 'test_data', '')
flags.DEFINE_string('HIGH_RES_IMAGE_FOLDER', 'images', '')
flags.DEFINE_string('AUDIO_FOLDER', 'audio', '')
flags.DEFINE_string('OUTPUT_FOLDER', 'output', '')
flags.DEFINE_string('STYLEGAN_CHECKPOINT', 'checkpoint/network-snapshot-030929.pkl', '')
flags.DEFINE_string('FUSION_CHECKPOINT', 'checkpoint/model-0', '')

# These variables are hard coded because they contain the names of different variable scopes of our encoders
LR_ENCODER_SCOPE = "LOW_RES_ENCODER"
AUDIO_ENCODER_SCOPE = "AUDIO_SPECTROGRAM_ENCODER"
FUSION_SCOPE = "AUDIO_VISUAL_FUSER"


create_dir(os.path.join(FLAGS.DATA_PATH, FLAGS.OUTPUT_FOLDER))

# Loading The List of Input High-Resolution Images and Corresponding Audio Tracks
image_pathes = sorted(glob.glob(os.path.join(FLAGS.DATA_PATH, FLAGS.HIGH_RES_IMAGE_FOLDER) + "/*.png"))
audio_pathes = sorted(glob.glob(os.path.join(FLAGS.DATA_PATH, FLAGS.AUDIO_FOLDER) + "/*.wav"))

# Defining Input Placeholders
image_path = tf.placeholder(tf.string)
audio_path = tf.placeholder(tf.string)

# Loading High-Resolution Image, Downsampled Low-Resolution Image, Preprocessed Audio, Nearest-Neighbor Interpolation of Inout Low-Resolution Image
high_res_image, low_res_image, audio, low_res_image_nearest = load_test_sample(image_path, audio_path)

# Constructing All Encoders
with tf.device("/GPU:0"):
    tflib.init_tf()

    _, _, G = pickle.load(open(FLAGS.STYLEGAN_CHECKPOINT, "rb"))

    Gs = tflib.Network(name=G.name, func_name="networks_stylegan.G_style", **G.static_kwargs)

    with tf.variable_scope(LR_ENCODER_SCOPE, reuse=tf.AUTO_REUSE):
        encoded_input = LowResEncoder(
            input=low_res_image,
            num_channels=3,
            resolution=8,
            batch_size=FLAGS.BATCH_SIZE,
            num_scales=3,
            n_filters=128,
            output_feature_size=512,
        )

    with tf.variable_scope(AUDIO_ENCODER_SCOPE, reuse=tf.AUTO_REUSE):
        audio_encoded_input = SpectrogramEncoder(
            input=audio,
            num_channels=1,
            resolution=257,
            batch_size=FLAGS.BATCH_SIZE,
            num_scales=6,
            n_filters=64,
            output_feature_size=512,
        )

    with tf.variable_scope(FUSION_SCOPE, reuse=tf.AUTO_REUSE):
        fused_mixed_input = FC3ResidualFuser(
            lr_input=tf.reshape(encoded_input, shape=[FLAGS.BATCH_SIZE, 6144]),
            audio_input=tf.reshape(audio_encoded_input, shape=[FLAGS.BATCH_SIZE, 6144]),
            batch_size=FLAGS.BATCH_SIZE,
        )

    low_res_encoded_image = tf.clip_by_value(
        revertZeroCenter(get_images(encoded_input, Gs, FLAGS.BATCH_SIZE)),
        clip_value_min=0,
        clip_value_max=1,
    )

    fused_image = tf.clip_by_value(
        revertZeroCenter(get_images(fused_mixed_input, Gs, FLAGS.BATCH_SIZE)),
        clip_value_min=0,
        clip_value_max=1,
    )

# Aggregating Variables of Different Encoders
G_vars = [v for v in tf.trainable_variables() if "G_" in v.name]

fusion_variables = (
    tf.contrib.framework.get_variables_to_restore(
        include=[AUDIO_ENCODER_SCOPE, LR_ENCODER_SCOPE, FUSION_SCOPE]
    )
    + G_vars
)

fusion_variables_init_fn = tf.contrib.framework.assign_from_checkpoint_fn(FLAGS.FUSION_CHECKPOINT, fusion_variables)

with tf.get_default_session() as sess:
    # Loading All Encoder Weights From Checkpoint
    fusion_variables_init_fn(sess)

    for im_path, a_path in zip(image_pathes, audio_pathes):
        (
            high_res_image_np,
            low_res_image_nearest_np,
            low_res_encoded_image_np,
            fused_image_np,
        ) = sess.run(
            [high_res_image, low_res_image_nearest, low_res_encoded_image, fused_image],
            feed_dict={image_path: im_path, audio_path: a_path},
        )

        # First Is The High-Resolution Image,
        # Second Is The Low-Resolution Input Image,
        # Third Is The Output of Low_Res_Encoder,
        # Fourth is The Result of Fusion
        imageio.imwrite(
            os.path.join(FLAGS.DATA_PATH,FLAGS.OUTPUT_FOLDER,os.path.basename(im_path)),
            np.hstack(
                [
                    high_res_image_np[0, :, :, :],
                    low_res_image_nearest_np[0, :, :, :],
                    low_res_encoded_image_np[0, :, :, :],
                    fused_image_np[0, :, :, :],
                ]
            ),
        )
