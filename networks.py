import tensorflow as tf

def HRImageEncoder(input, num_channels=3,
                   resolution=128, batch_size=0, num_scales=6, nonlinearity='lrelu', normalizer='instance_norm',
                   n_filters=64,
                   components=None,
                   is_template_graph=False):
    input.set_shape([batch_size, resolution, resolution, num_channels])

    if nonlinearity == 'lrelu':
        hidden_activation = tf.nn.leaky_relu

    if normalizer == 'instance_norm':
        normalizer_fn = tf.contrib.layers.instance_norm

    with tf.variable_scope('', reuse=tf.AUTO_REUSE):

        # IMAGE Conv NET Branch
        convs = [input]

        for i in range(num_scales):

            if i == 0:
                convs.append(tf.contrib.layers.conv2d(convs[-1], activation_fn=hidden_activation,
                                                      num_outputs=n_filters,
                                                      kernel_size=4,
                                                      stride=1,
                                                      padding='SAME',
                                                      normalizer_fn=None))
            else:
                convs.append(tf.contrib.layers.conv2d(convs[-1], activation_fn=hidden_activation,
                                                      num_outputs=n_filters,
                                                      kernel_size=4,
                                                      stride=1,
                                                      padding='SAME',
                                                      normalizer_fn=normalizer_fn))

            n_filters = n_filters * 2

            if n_filters > 1024:
                n_filters = 1024

            if i != (num_scales - 1):
                convs.append(tf.contrib.layers.conv2d(convs[-1], activation_fn=hidden_activation,
                                                      num_outputs=n_filters,
                                                      kernel_size=4,
                                                      stride=2,
                                                      padding='SAME',
                                                      normalizer_fn=normalizer_fn))

        convs.append(tf.contrib.layers.flatten(convs[-1]))

        convs.append(tf.contrib.layers.fully_connected(
            inputs=convs[-1],
            num_outputs=6144,
            activation_fn=None,
            normalizer_fn=None))

        convs.append(tf.reshape(convs[-1], shape=[convs[-1].get_shape().as_list()[0], 12, 512]))

        return convs[-1]


def SpectrogramEncoder(input, num_channels=1,
                       resolution=129, batch_size=0, num_scales=6, nonlinearity='lrelu',
                       n_filters=64, output_feature_size=0, max_filters=2048,
                       components=None,
                       is_template_graph=False):
    input.set_shape([batch_size, resolution, resolution, num_channels])

    if nonlinearity == 'lrelu':
        hidden_activation = tf.nn.leaky_relu

    normalizer_fn = None

    with tf.variable_scope('', reuse=tf.AUTO_REUSE):

        # IMAGE Conv NET Branch
        convs = [input]

        convs.append(tf.contrib.layers.conv2d(convs[-1], activation_fn=hidden_activation,
                                              num_outputs=n_filters,
                                              kernel_size=4,
                                              stride=2,
                                              padding='SAME',
                                              normalizer_fn=None))

        for i in range(num_scales):

            convs.append(tf.contrib.layers.conv2d(convs[-1], activation_fn=hidden_activation,
                                                  num_outputs=n_filters,
                                                  kernel_size=4,
                                                  stride=1,
                                                  padding='SAME',
                                                  normalizer_fn=normalizer_fn))

            n_filters = n_filters * 2

            if n_filters > max_filters:
                n_filters = max_filters

            convs.append(tf.contrib.layers.conv2d(convs[-1], activation_fn=hidden_activation,
                                                  num_outputs=n_filters,
                                                  kernel_size=4,
                                                  stride=2,
                                                  padding='SAME',
                                                  normalizer_fn=normalizer_fn))

        convs.append(tf.contrib.layers.flatten(convs[-1]))

        convs.append(tf.contrib.layers.fully_connected(
            inputs=convs[-1],
            num_outputs=8192,
            activation_fn=hidden_activation,
            normalizer_fn=None))

        convs.append(tf.contrib.layers.fully_connected(
            inputs=convs[-1],
            num_outputs=12 * output_feature_size,
            activation_fn=None,
            normalizer_fn=None))

        convs.append(tf.reshape(convs[-1], shape=[convs[-1].get_shape().as_list()[0], 12, output_feature_size]))

        return convs[-1]


def LowResEncoder(input, num_channels=3,
                  resolution=128, batch_size=0, num_scales=6, nonlinearity='lrelu', normalizer='instance_norm',
                  n_filters=128, output_feature_size=0,
                  components=None,
                  is_template_graph=False):
    input.set_shape([batch_size, resolution, resolution, num_channels])

    if nonlinearity == 'lrelu':
        hidden_activation = tf.nn.leaky_relu

    if normalizer == 'instance_norm':
        normalizer_fn = tf.contrib.layers.instance_norm

    with tf.variable_scope('', reuse=tf.AUTO_REUSE):

        # IMAGE Conv NET Branch
        convs = [input]

        convs.append(tf.contrib.layers.conv2d(convs[-1], activation_fn=hidden_activation,
                                              num_outputs=n_filters,
                                              kernel_size=3,
                                              stride=1,
                                              padding='SAME',
                                              normalizer_fn=None))

        for i in range(num_scales):

            convs.append(tf.contrib.layers.conv2d(convs[-1], activation_fn=hidden_activation,
                                                  num_outputs=n_filters,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding='SAME',
                                                  normalizer_fn=normalizer_fn))

            convs.append(tf.contrib.layers.conv2d(convs[-1], activation_fn=hidden_activation,
                                                  num_outputs=n_filters,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding='SAME',
                                                  normalizer_fn=normalizer_fn))

            n_filters = n_filters * 3

            if n_filters > 1024:
                n_filters = 1024
            if i != num_scales - 1:
                convs.append(tf.contrib.layers.conv2d(convs[-1], activation_fn=hidden_activation,
                                                      num_outputs=n_filters,
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding='SAME',
                                                      normalizer_fn=normalizer_fn))

        convs.append(tf.contrib.layers.flatten(convs[-1]))

        convs.append(tf.contrib.layers.fully_connected(
            inputs=convs[-1],
            num_outputs=12 * output_feature_size,
            activation_fn=hidden_activation,
            normalizer_fn=None))

        convs.append(tf.contrib.layers.fully_connected(
            inputs=convs[-1],
            num_outputs=12 * output_feature_size,
            activation_fn=None,
            normalizer_fn=None))

        convs.append(tf.reshape(convs[-1], shape=[convs[-1].get_shape().as_list()[0], 12, output_feature_size]))

        return convs[-1]

def FC3ResidualFuser(lr_input, audio_input, batch_size=0,
                          components=None,
                          is_template_graph=False):
    lr_input.set_shape([batch_size, 6144])
    audio_input.set_shape([batch_size, 6144])

    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        # IMAGE Conv NET Branch
        convs = [tf.concat([lr_input, audio_input], axis=1)]

        convs.append(tf.contrib.layers.fully_connected(
            inputs=convs[-1],
            num_outputs=6144,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=None))

        convs.append(tf.contrib.layers.fully_connected(
            inputs=convs[-1],
            num_outputs=6144,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=None))

        convs.append(tf.contrib.layers.fully_connected(
            inputs=convs[-1],
            num_outputs=6144,
            activation_fn=None,
            normalizer_fn=None))

        convs.append(1.0*convs[-1] + lr_input)

        convs.append(tf.reshape(convs[-1], shape=[convs[-1].get_shape().as_list()[0], 12, 512]))

        return convs[-1]

