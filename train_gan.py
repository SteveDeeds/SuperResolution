import tensorflow as tf
import numpy as np
import cv2
import glob
import os
import random

import gan_model as model_def

kEpochCount = 1
kMaxImages = 10
kTrainingPatchesPerImage = 100
kBatchSize = 100
kCycleCount = 100


def getFusedModel():
    model = model_def.getFusedModel()
    model.summary()
    print("output of model: " + str(model.output.shape))
    return model


def getDiscriminatorModel():
    model = model_def.getDiscriminator()
    model.summary()
    print("output of model: " + str(model.output.shape))
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


def trainFusedModel(model, inputs):
    model_def.freezeDiscriminator()
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "fused_checkpoint.hdf5", monitor='val_loss', verbose=1,
        save_best_only=False, save_weights_only=False)

    inputTensor = tf.constant(inputs, 'float32')
    outputTensor = tf.constant([1] * len(inputs), 'float32')

    history = model.fit(
        x=inputTensor, y=outputTensor,
        epochs=kEpochCount, batch_size=kBatchSize,
        callbacks=[checkpoint], verbose=1)
    losses = history.history['loss']
    return losses[len(losses)-1]


def trainGeneratorModel(model, inputs, truths):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "generator_checkpoint.hdf5", monitor='val_loss', verbose=1,
        save_best_only=False, save_weights_only=False)

    inputTensor = tf.constant(inputs, 'float32')
    outputTensor = tf.constant(truths, 'float32')

    history = model.fit(
        x=inputTensor, y=outputTensor,
        epochs=kEpochCount, batch_size=kBatchSize,
        callbacks=[checkpoint], verbose=1)
    losses = history.history['loss']
    return losses[len(losses)-1]


def trainDiscriminatorModel(discriminator, generator, inputs, truths):
    model_def.thawDiscriminator()
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "discriminator_checkpoint.hdf5", monitor='val_loss', verbose=1,
        save_best_only=False, save_weights_only=False)

    # First, evaluate the generator to produce the images to discriminate
    paintings = generator.predict(inputs)
    ones = np.array([1] * len(truths), 'float32')
    zeros = np.array([0] * len(paintings), 'float32')
    outputs = np.concatenate([ones, zeros], axis=0)

    bothInputs = np.concatenate([truths, paintings], axis=0)

    inputTensor = tf.constant(bothInputs, 'float32')
    outputTensor = tf.constant(outputs, 'float32')

    history = discriminator.fit(
        x=inputTensor, y=outputTensor,
        epochs=kEpochCount, batch_size=kBatchSize,
        callbacks=[checkpoint], verbose=1)
    losses = history.history['loss']
    return losses[len(losses)-1]


def evalModel(generator, inputs, truths):
    print("Inputs shape: " + str(inputs.shape))
    print("First shape: " + str(np.array(inputs[0]).shape))

    num_outputs = min(5, len(inputs))
    indexes = random.sample(range(0, inputs.shape[0]), num_outputs)
    for index in indexes:
        outputs = generator.predict(inputs)

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
    size = min(original.shape[0], original.shape[1])
    original = original[0:size][0:size]
    inputSize = 256 + 2 * model_def.kInputPadding
    original = cv2.resize(original, (inputSize, inputSize),
                          interpolation=cv2.INTER_LINEAR)
    blurry = getBlurryImage(original)
    print("Original shape: " + str(original.shape))
    print("Blurry shape: " + str(blurry.shape))

    (newIns, newOuts) = getTrainingPatches(original, blurry,
                                           kTrainingPatchesPerImage)
    return (np.append(inputs, newIns, axis=0),
            np.append(outputs, newOuts, axis=0))


def main():
    generator = model_def.getGeneratorModel()
    discriminator = model_def.getDiscriminator()
    fused = model_def.getFusedModel()

    inputs = np.empty((0, model_def.kInputPatchSize,
                       model_def.kInputPatchSize, 3))
    truths = np.empty((0, model_def.kOutputPatchSize,
                       model_def.kOutputPatchSize, 3))

    fileNames = glob.glob(os.path.join("input", "*.jpg"))
    fileNames.extend(glob.glob(os.path.join("input", "*.png")))
    i = 0
    for fname in fileNames:
        (inputs, truths) = addIOFromImage(fname, inputs, truths)
        i += 1
        if i >= kMaxImages:
            break

    print("Inputs shape: " + str(inputs.shape))

    for i in range(kCycleCount):
        print("Generative cycle " + str(i) + " of " + str(kCycleCount))
        trainGeneratorModel(generator, inputs, truths)

    fusedError = 1.0
    discriminatorError = 1.0

    for i in range(kCycleCount):
        print("discriminator: " + str(discriminatorError) +
              " fused: " + str(fusedError))
        if (discriminatorError > fusedError):
            discriminatorError = trainDiscriminatorModel(
                discriminator, generator, inputs, truths)
        else:
            fusedError = trainFusedModel(fused, inputs)

    # TODO: Use different images for evaluation
    evalModel(generator, inputs, truths)


if __name__ == "__main__":
    main()
