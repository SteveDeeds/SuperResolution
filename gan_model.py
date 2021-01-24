import tensorflow as tf
from tensorflow.keras import backend as K

kInputPadding = 1
kScaleFactor = 4
kOutputPatchSize = 64
kInputPatchSize = kOutputPatchSize + 2 * kInputPadding


def my_psnr(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return -100.0
    max_pixel = 1.0
    psnr = 20 * K.log(max_pixel / K.sqrt(mse))
    return -psnr
    # #difference between true label and predicted label
    # error = y_true-y_pred
    # #square of the error
    # sqr_error = K.square(error)
    # #mean of the square of the error
    # mean_sqr_error = K.mean(sqr_error)
    # #square root of the mean of the square of the error
    # sqrt_mean_sqr_error = K.sqrt(mean_sqr_error)
    # #return the error
    # return sqrt_mean_sqr_error


generatorLayers = []


def addGenerator(model):
    print("Creating generator")
    if len(generatorLayers) == 0:
        generatorLayers.append(
            tf.keras.layers.AveragePooling2D(pool_size=(4, 4)))
        generatorLayers.append(tf.keras.layers.Conv2DTranspose(
            3, 4, strides=4, activation='hard_sigmoid',
            kernel_initializer='zeros',
            bias_initializer='zeros'))

    for gl in generatorLayers:
        model.add(gl)


def getFusedModel():
    model = tf.keras.Sequential()
    # with filter sizes of 9, 3, and 5 the input needs to be 270x270 for a 256x256 output 256 + 9 - 1 + 3 -1 + 5 - 1 = 270
    model.add(tf.keras.layers.Input(
        shape=(kInputPatchSize, kInputPatchSize, 3)))

    addGenerator(model)
    addDiscriminator(model)

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(
        0.003), metrics=['mean_squared_error', 'accuracy'])
    print("Fused")
    model.summary()
    return model


def getGeneratorModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(
        shape=(kInputPatchSize, kInputPatchSize, 3)))
    addGenerator(model)
    # Best to train the generator on expected images first before applying GAN
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(
        0.003), metrics=['mean_squared_error', 'accuracy'])
    print("Generator")
    model.summary()
    return model


discriminatorLayers = []


def addDiscriminator(model):
    if len(discriminatorLayers) == 0:
        discriminatorLayers.append(tf.keras.layers.Conv2D(
            3, 3, activation='linear', padding='valid'))
        discriminatorLayers.append(
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))
        discriminatorLayers.append(tf.keras.layers.Conv2D(
            3, 3, activation='linear', padding='valid'))
        discriminatorLayers.append(
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))
        discriminatorLayers.append(
            tf.keras.layers.Flatten())
        discriminatorLayers.append(
            tf.keras.layers.Dense(units=1, activation='relu'))

    for dl in discriminatorLayers:
        model.add(dl)


def freezeDiscriminator():
    for l in discriminatorLayers:
        l.trainable = False


def thawDiscriminator():
    for l in discriminatorLayers:
        l.trainable = True


def getDiscriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(
        shape=(kOutputPatchSize, kOutputPatchSize, 3)))

    addDiscriminator(model)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(
        0.003), metrics=['mean_squared_error', 'accuracy'])

    print("Discriminator")
    model.summary()
    return model
