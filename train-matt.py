import tensorflow as tf
import numpy as np
import cv2
import glob
import os
import random

import bigger_model as model_def


def getModel():
    model = model_def.getModel()
    model.summary()
    assert(model.output.shape[1] == model_def.kOutputPatchSize)
    return model


def getBlurryImage(image):
    height = image.shape[1]
    width = image.shape[0]
    small = cv2.resize(image, dsize=(
        int(height / model_def.kScaleFactor),
        int(width / model_def.kScaleFactor)))
    blurry = cv2.resize(small, dsize=(height, width))
    return blurry


def getPatch(image, x1, y1, size):
    x2 = x1 + size
    y2 = y1 + size
    return image[x1:x2, y1:y2, 0:3]


def getIOPair(sharp, blurry):
    width = blurry.shape[0]
    height = blurry.shape[1]
    x1 = np.random.randint(0, width - model_def.kInputPatchSize)
    y1 = np.random.randint(0, height - model_def.kInputPatchSize)
    blurryPatch = getPatch(blurry, x1, y1, model_def.kInputPatchSize)
    sharpPatch = getPatch(
        sharp,
        x1 + model_def.kInputPadding,
        y1 + model_def.kInputPadding,
        model_def.kOutputPatchSize)
    return (blurryPatch, sharpPatch)


def getTrainingPatches(sharp, blurry, count):
    inputPatches = np.empty(
        shape=(0, model_def.kInputPatchSize, model_def.kInputPatchSize, 3), dtype='float32')
    outputPatches = np.empty(
        shape=(0, model_def.kOutputPatchSize, model_def.kOutputPatchSize, 3), dtype='float32')
    for index in range(0, count):
        (i, o) = getIOPair(sharp, blurry)
        inputPatches = np.append(inputPatches, [i], axis=0)
        outputPatches = np.append(outputPatches, [o], axis=0)
    return (inputPatches, outputPatches)


def getCoveringPatches(blurry, patchSize):
    patches = np.empty(
        shape=(0, patchSize, patchSize, 3), dtype='float32')
    width = blurry.shape[0]
    height = blurry.shape[1]
    x1 = 0
    while True:
        x2 = x1 + patchSize
        y1 = 0
        while True:
            y2 = y1 + patchSize
            blurryPatch = blurry[x1:x2, y1:y2, 0:3]
            patches = np.append(patches, [blurryPatch], axis=0)
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
            dsize=(model_def.kOutputPatchSize, model_def.kOutputPatchSize),
            interpolation=cv2.INTER_NEAREST)
        outputImg = np.array(outputs[index, :, :, :]) * 255
        truthImg = truths[index, :, :, :] * 255
        compositeImg = np.append(inputImg, outputImg, axis=1)
        compositeImg = np.append(compositeImg, truthImg, axis=1)

        cv2.imwrite(outputFile, compositeImg)


def addIOFromImage(filename, inputs, outputs):
    original = cv2.imread(
        os.path.join(filename)) / 255
    blurry = getBlurryImage(original)
    print("Original shape: " + str(original.shape))
    print("Blurry shape: " + str(blurry.shape))

    (newIns, newOuts) = getTrainingPatches(original, blurry, 10)
    return (np.append(inputs, newIns, axis=0),
            np.append(outputs, newOuts, axis=0))


def main():
    model = getModel()

    inputs = np.empty((0, model_def.kInputPatchSize,
                       model_def.kInputPatchSize, 3))
    outputs = np.empty((0, model_def.kOutputPatchSize,
                        model_def.kOutputPatchSize, 3))

    fileNames = glob.glob(os.path.join("input", "*.jpg"))
    fileNames.extend(glob.glob(os.path.join("input", "*.png")))
    for fname in fileNames:
        (inputs, outputs) = addIOFromImage(fname, inputs, outputs)

    print("Inputs shape: " + str(inputs.shape))

    trainModel(model, inputs, outputs)
    # TODO: Use different images for evaluation
    evalModel(model, inputs, outputs)


if __name__ == "__main__":
    main()
