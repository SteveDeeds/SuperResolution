import tensorflow as tf
import numpy as np
import cv2
import glob
import os
import random

kInputPadding = 4
kScaleFactor = 4
kOutputPatchSize = 256

kInputPatchSize = int(kOutputPatchSize / kScaleFactor)


def getModel():
    model = tf.keras.Sequential()
    inputShape = (kInputPatchSize, kInputPatchSize, 3)
    print("Input shape: " + str(inputShape))
    # See section 6 here:
    # https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
    model.add(tf.keras.layers.Conv2DTranspose(
        3,  # Number of filters
        13,
        padding='same',
        output_padding=kScaleFactor - 1,
        strides=kScaleFactor,
        activation='linear',
        input_shape=inputShape,
        kernel_initializer='zeros'))
    print("Output shape: " + str(model.output.shape))
    assert(model.output.shape[1] == kOutputPatchSize)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(
        0.01), metrics=['mean_squared_error', 'accuracy'])
    model.summary()
    return model


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
    width = sharp.shape[0] / kScaleFactor
    height = sharp.shape[1] / kScaleFactor
    x1 = np.random.randint(0, width - kInputPatchSize)
    y1 = np.random.randint(0, height - kInputPatchSize)
    sharpPatch = getPatch(sharp,
                          x1 * kScaleFactor, y1 * kScaleFactor, kOutputPatchSize)
    smallPatch = getPatch(small, x1, y1, kInputPatchSize)
    return (smallPatch, sharpPatch)


def getTrainingPatches(sharp, small, count):
    inputPatches = np.empty(
        shape=(0, kInputPatchSize, kInputPatchSize, 3), dtype='float32')
    outputPatches = np.empty(
        shape=(0, kOutputPatchSize, kOutputPatchSize, 3), dtype='float32')
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

    (newIns, newOuts) = getTrainingPatches(original, small, 50)
    return (np.append(inputs, newIns, axis=0),
            np.append(outputs, newOuts, axis=0))


def main():
    model = getModel()

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
