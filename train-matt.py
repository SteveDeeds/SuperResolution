import tensorflow as tf
import numpy as np
import cv2
import glob
import os


def getModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
        1, 13, padding='same',
        activation='linear', input_shape=(63, 63, 1)))
    print("Output shape: " + str(model.output.shape))
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(
        0.05), metrics=['mean_squared_error', 'accuracy'])
    model.summary()
    return model


def getBlurryImage(image):
    height = image.shape[1]
    width = image.shape[0]
    small = cv2.resize(image, dsize=(int(height / 4), int(width / 4)))
    blurry = cv2.resize(small, dsize=(height, width),
                        interpolation=cv2.INTER_CUBIC)
    return blurry


def getIOPair(sharp, blurry):
    width = sharp.shape[0]
    height = sharp.shape[1]
    x1 = np.random.randint(0, width - 63)
    y1 = np.random.randint(0, height - 63)
    x2 = x1 + 63
    y2 = y1 + 63
    sharpPatch = sharp[x1:x2, y1:y2, 0:1]
    blurryPatch = blurry[x1:x2, y1:y2, 0:1]
    return (blurryPatch, sharpPatch)


def getTrainingPatches(sharp, blurry, count):
    inputPatches = np.empty(shape=(0, 63, 63, 1), dtype='float32')
    outputPatches = np.empty(shape=(0, 63, 63, 1), dtype='float32')
    for index in range(0, count):
        (i, o) = getIOPair(sharp, blurry)
        inputPatches = np.append(inputPatches, [i], axis=0)
        outputPatches = np.append(outputPatches, [o], axis=0)
    return (inputPatches, outputPatches)


def getCoveringPatches(blurry):
    patches = np.empty(shape=(0, 63, 63, 1), dtype='float32')
    width = blurry.shape[0]
    height = blurry.shape[1]
    x1 = 0
    while True:
        x2 = x1 + 63
        y1 = 0
        while True:
            y2 = y1 + 63
            blurryPatch = blurry[x1:x2, y1:y2, 0:1]
            patches = np.append(patches, [blurryPatch], axis=0)
            y1 = y2
            if (y1 + 63 > height):
                break
        x1 = x2
        if (x1 + 63 > width):
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
    for index in range(inputs.shape[0]):
        outputs = model.apply(inputs)

        outputFile = os.path.join("eval", "comaprison-"+str(index)+".png")
        print("Output file: " + outputFile)
        inputImg = inputs[index, :, :, :] * 255
        outputImg = np.array(outputs[index, :, :, :]) * 255
        truthImg = truths[index, :, :, :] * 255
        compositeImg = np.append(inputImg, outputImg, axis=1)
        compositeImg = np.append(compositeImg, truthImg, axis=1)

        cv2.imwrite(outputFile, compositeImg)


def main():
    original = cv2.imread("half-dome.png") / 255
    blurry = getBlurryImage(original)
    cv2.imwrite("half-dome-blurry.png", blurry * 255)

    print("Original shape: " + str(original.shape))
    print("Blurry shape: " + str(blurry.shape))

    (inputs, outputs) = getTrainingPatches(original, blurry, 1000)
    print("Inputs shape: " + str(inputs.shape))

    model = getModel()
    trainModel(model, inputs, outputs)

    truths = getCoveringPatches(original)
    covering = getCoveringPatches(blurry)
    evalModel(model, covering, truths)


if __name__ == "__main__":
    main()
