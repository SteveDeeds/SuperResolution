import tensorflow as tf
import numpy as np
import cv2
import glob
import os
import random
from tensorflow.keras import backend as K

import smaller_model as model_def


def getModel():
    model = model_def.getModel()
    assert(model.output.shape[1] == model_def.kOutputPatchSize)
    model.summary()
    return model


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


def getSmallImage(image):
    height = image.shape[1]
    width = image.shape[0]
    small = cv2.resize(image, dsize=(
        int(height / kScaleFactor), int(width / kScaleFactor)))
    return small


def getPatch(image, x1, y1, size):
    x2 = x1 + size
    y2 = y1 + size
    return image[x1:x2, y1:y2, 0:3]


def getIOPair(sharp, small):
    width = small.shape[0]
    height = small.shape[1]
    x1 = np.random.randint(0, width - model_def.kInputPatchSize)
    y1 = np.random.randint(0, height - model_def.kInputPatchSize)
    smallPatch = getPatch(small, x1, y1, model_def.kInputPatchSize)
    sharpPatch = getPatch(
        sharp,
        (x1 + model_def.kInputPadding) * model_def.kScaleFactor,
        (y1 + model_def.kInputPadding) * model_def.kScaleFactor, model_def.kOutputPatchSize)
    return (smallPatch, sharpPatch)


def getTrainingPatches(sharp, small, count):
    inputPatches = np.empty(
        shape=(0, model_def.kInputPatchSize, model_def.kInputPatchSize, 3), dtype='float32')
    outputPatches = np.empty(
        shape=(0, model_def.kOutputPatchSize, model_def.kOutputPatchSize, 3), dtype='float32')
    for index in range(0, count):
        (i, o) = getIOPair(sharp, small)
        inputPatches = np.append(inputPatches, [i], axis=0)
        outputPatches = np.append(outputPatches, [o], axis=0)
    return (inputPatches, outputPatches)


def getCoveringPatches(small, patchSize):
    patches = np.empty(
        shape=(0, patchSize, patchSize, 3), dtype='float32')
    width = small.shape[0]
    height = small.shape[1]
    x1 = 0
    while True:
        x2 = x1 + patchSize
        y1 = 0
        while True:
            y2 = y1 + patchSize
            smallPatch = small[x1:x2, y1:y2, 0:3]
            patches = np.append(patches, [smallPatch], axis=0)
            y1 = y2
            if (y1 + patchSize > height):
                break
        x1 = x2
        if (x1 + patchSize > width):
            break
    return patches


def trainModel(model, inputs, outputs):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "check_{epoch}.hdf5", monitor='val_loss', verbose=1,
        save_best_only=False, save_weights_only=False)

    inputTensor = tf.constant(inputs, 'float32')
    outputTensor = tf.constant(outputs, 'float32')

    print("Input shape: " + str(inputTensor.shape))
    print("Output shape: " + str(outputTensor.shape))

    history = model.fit(
        x=inputTensor, y=outputTensor,
        epochs=10, batch_size=100, callbacks=[checkpoint], verbose=1)
    print(history.history['loss'])


def evalModel(model, inputs, truths):
    print("Inputs shape: " + str(inputs.shape))
    print("First shape: " + str(np.array(inputs[0]).shape))

    indexes = random.sample(range(0, inputs.shape[0]), 5)
    for index in indexes:
        outputs = model.apply(inputs)

        outputFile = os.path.join("eval", "comaprison-"+str(index)+".png")
        print("Output file: " + outputFile)
        inputImg = cv2.resize(
            inputs[index, :, :, :] * 255,
            dsize=(kOutputPatchSize, kOutputPatchSize),
            interpolation=cv2.INTER_NEAREST)
        outputImg = np.array(outputs[index, :, :, :]) * 255
        truthImg = truths[index, :, :, :] * 255
        compositeImg = np.append(inputImg, outputImg, axis=1)
        compositeImg = np.append(compositeImg, truthImg, axis=1)

        cv2.imwrite(outputFile, compositeImg)


def addIOFromImage(filename, inputs, outputs):
    original = cv2.imread(
        os.path.join(filename)) / 255
    small = getSmallImage(original)
    print("Original shape: " + str(original.shape))
    print("Small shape: " + str(small.shape))

    (newIns, newOuts) = getTrainingPatches(original, small, 10)
    return (np.append(inputs, newIns, axis=0),
            np.append(outputs, newOuts, axis=0))


def main():
    model = getBiggerModel()

    inputs = np.empty((0, kInputPatchSize, kInputPatchSize, 3))
    outputs = np.empty((0, kOutputPatchSize, kOutputPatchSize, 3))

    fileNames = glob.glob(os.path.join("input", "*.jpg"))
    for fname in fileNames:
        (inputs, outputs) = addIOFromImage(fname, inputs, outputs)

    print("Inputs shape: " + str(inputs.shape))

    trainModel(model, inputs, outputs)
    # TODO: Use different images for evaluation
    evalModel(model, inputs, outputs)


if __name__ == "__main__":
    main()
