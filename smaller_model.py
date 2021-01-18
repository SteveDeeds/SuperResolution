import tensorflow as tf

kInputPadding = 6
kScaleFactor = 4
kOutputPatchSize = 64
kInputPatchSize = kOutputPatchSize + 2 * kInputPadding


def getModel():
    model = tf.keras.Sequential()
    inputShape = (kInputPatchSize, kInputPatchSize, 3)
    print("Input shape: " + str(inputShape))
    # See section 6 here:
    # https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
    model.add(tf.keras.layers.Conv2D(
        3,  # Number of filters
        13,
        padding='valid',
        strides=1,
        activation='linear',
        input_shape=inputShape,
        kernel_initializer='zeros'))
    print("Output shape: " + str(model.output.shape))
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(
        0.01), metrics=['mean_squared_error', 'accuracy'])
    return model
