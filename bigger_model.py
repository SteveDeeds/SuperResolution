import tensorflow as tf
from tensorflow.keras import backend as K

kInputPadding = 7
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


def getModel():
    model = tf.keras.Sequential()
    # with filter sizes of 9, 3, and 5 the input needs to be 270x270 for a 256x256 output 256 + 9 - 1 + 3 -1 + 5 - 1 = 270
    model.add(tf.keras.layers.Input(
        shape=(kInputPatchSize, kInputPatchSize, 3)))
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
