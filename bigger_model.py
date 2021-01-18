kInputPadding = 4
kScaleFactor = 4
kOutputPatchSize = 256

kInputPatchSize = int(kOutputPatchSize / kScaleFactor)


def getBiggerModel():
    model = tf.keras.Sequential()
    # with filter sizes of 9, 3, and 5 the input needs to be 270x270 for a 256x256 output 256 + 9 - 1 + 3 -1 + 5 - 1 = 270
    model.add(tf.keras.layers.Input(shape=(270, 270, 3)))
    model.add(tf.keras.layers.Conv2D(
        128, 9, activation='relu', padding='valid'))
    model.add(tf.keras.layers.Conv2D(
        64, 3, activation='relu', padding='valid'))
    model.add(tf.keras.layers.Conv2D(
        3, 5, activation='hard_sigmoid', padding='valid'))
    model.compile(loss=my_psnr, optimizer=tf.keras.optimizers.Adam(
        0.001), metrics=['mean_squared_error', 'accuracy'])
    model.summary()
    return model
