###############################################
# Requires two user provided files:
# photo.jpg (a photo to start from)
# painting.png (a painting to use for a target)
# (yes, one is a png, the other is a jpg.  I know that's weird)
# the discriminator and the generator are very simple to make it run fast.  
# model layers will need to be uncommented and the kPadding changed to create more interesting results.
###############################################

import glob
import math
import os
import random

import cv2
import numpy as np
import tensorflow as tf

import train_matt

kPatchSize = 32
kPadding = 2
kPatchesPerEpoch = 10

kInputPatchSize = kPatchSize+2*kPadding

def getPatch(img, patchSize=kPatchSize):
    (width, height,_) = img.shape
    x1 = np.random.randint(0, width - patchSize)
    y1 = np.random.randint(0, height - patchSize)
    x2 = x1 + patchSize
    y2 = y1 + patchSize
    return img[x1:x2, y1:y2, 0:3]

def getPatchPair(img, outputPatchSize, padding):
    (width, height,_) = img.shape
    x1 = np.random.randint(padding, width - (outputPatchSize+2*padding))
    y1 = np.random.randint(padding, height - (outputPatchSize+2*padding))
    x2 = x1 + outputPatchSize
    y2 = y1 + outputPatchSize
    inputImg = img[x1:x2, y1:y2, 0:3]
    outputImg = img[x1-padding:x2+padding, y1-padding:y2+padding, 0:3]

    return inputImg,outputImg

def getDiscriminatorModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(kPatchSize, kPatchSize, 3)))
    #model.add(tf.keras.layers.Conv2D(64, 9, strides=1, activation='relu'))
    #model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(32, 9, strides=1, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation = 'linear'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mean_squared_error', 'accuracy'])
    print("Discriminator")
    model.summary()
    return model

generatorLayers = []
def getGeneratorModel(inputShape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(inputShape)))
    global generatorLayers
    generatorLayers = []
    #generatorLayers.append(tf.keras.layers.Conv2D(128, 13, strides=1, padding = 'valid', activation='relu'))
    #generatorLayers.append(tf.keras.layers.Conv2D(32, 9, strides=1, padding = 'valid', activation='relu'))
    generatorLayers.append(tf.keras.layers.Conv2D(3, 5, strides=1, padding = 'valid', activation='relu'))
    for gl in generatorLayers:
        model.add(gl)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mean_squared_error', 'accuracy'])
    print("Generator")
    model.summary()
    return model

def getFusedModel(generator, discriminator):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(kPatchSize+2*kPadding, kPatchSize+2*kPadding, 3)))
    for gl in generatorLayers:
        model.add(gl)
    discriminator.trainable = False
    model.add(discriminator)

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(
        0.003), metrics=['mean_squared_error', 'accuracy'])
    print("Fused")
    model.summary()
    return model    

def trainModel(model, inputs, outputs, epochs =1, name =""):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        name + "_checkpoint.hdf5", monitor='val_loss', verbose=0,
        save_best_only=True, save_weights_only=False)

    inputTensor = tf.constant(inputs, 'float32')
    outputTensor = tf.constant(outputs, 'float32')

    #print("Input shape: " + str(inputTensor.shape))
    #print("Output shape: " + str(outputTensor.shape))

    history = model.fit(
        x=inputTensor, y=outputTensor,
        epochs=epochs, batch_size=10,
        callbacks=[checkpoint], verbose=1,
        validation_split=0.2)
    accuracy = history.history['accuracy']
    return accuracy[-1]

def evalModel(model, inputs):
    inputTensor = tf.constant(inputs, 'float32')
    outputs = model.predict(inputTensor)
    print(outputs)

def imageFromPatches(patches, verticalPatchCount):
    resultImage = np.empty(
        (0, kPatchSize * verticalPatchCount, 3), dtype='float32')

    i = 0
    while (i < len(patches)):
        resultLine = np.empty(
            (kPatchSize, 0, 3), dtype='float32')
        for y in range(0, verticalPatchCount):
            if i >= len(patches):
                break
            resultLine = np.append(resultLine, patches[i], axis=1)
            i += 1
        resultImage = np.append(resultImage, resultLine, axis=0)

    return resultImage

def main():

    painting = cv2.imread("painting.png") / 255
    photo = cv2.imread("photo.jpg") / 255

   # train generator to replicate the input as a starting point
    inputs = []
    outputs = []
    for count in range(1000):
        smaller, bigger = getPatchPair(photo,kPatchSize,kPadding)
        inputs.append(bigger)
        outputs.append(smaller)
    generatorModel = getGeneratorModel(inputs[0].shape)
    trainModel(generatorModel,inputs,outputs,epochs=10, name="generator")

    # create a painting from the photo
    patches = train_matt.getCoveringPatches(photo, kInputPatchSize, kPadding)
    forgeries = generatorModel.predict(tf.constant(patches, 'float32'))

    verticalPatchCount = int(math.floor((photo.shape[1]-kPadding*2)/ kPatchSize))
    cv2.imwrite("output.jpg", imageFromPatches(forgeries,verticalPatchCount)*255)

    # train the discriminator to tell the difference between photos and paintings.
    inputs = []
    outputs = []
    for count in range(1000):
        if random.random()>0.5 :
            inputs.append(getPatch(painting))
            outputs.append(1)
        else:
            inputs.append(getPatch(photo))
            outputs.append(0)
    discriminatorModel = getDiscriminatorModel()
    trainModel(discriminatorModel,inputs,outputs,epochs=10, name="discriminator")

    discriminatorAccuracy = 0
    itteration = 0
    oldForgeries = []
    for loops in range(1000):
        # train the generator portion of the fused model
        fusedAccuracy = 0
        epochs = 1
        fusedModel = getFusedModel(generatorModel, discriminatorModel)
        while fusedAccuracy < 0.6:
            inputs = []
            outputs = []
            for count in range(kPatchesPerEpoch):
                inputs.append(getPatch(photo, patchSize=kPatchSize+2*kPadding))
                outputs.append(1)
            print("Train fused model.")
            # if the model has gotten lost, start over.
            if epochs > 10:
                generatorModel = getGeneratorModel(inputs[0].shape)
                fusedModel = getFusedModel(generatorModel, discriminatorModel)
            fusedAccuracy = trainModel(fusedModel,inputs,outputs, name="fused",epochs=epochs)
            print("Fused Acuracy = " + str(fusedAccuracy))
            epochs = epochs + 1

        # create a painting from the photo
        patches = train_matt.getCoveringPatches(photo, kInputPatchSize, kPadding)
        outputPatches = generatorModel.predict(tf.constant(patches, 'float32'))

        verticalPatchCount = int(math.floor((photo.shape[1]-kPadding*2)/ kPatchSize))
        cv2.imwrite("output"+str(itteration)+".jpg", imageFromPatches(outputPatches,verticalPatchCount)*255)
        itteration = itteration + 1

        # train the discriminator from the output of the generator
        discriminatorAccuracy = 0
        epochs = 1
        while discriminatorAccuracy < 0.8:
            inputs = []
            for count in range(100):
                inputs.append(getPatch(painting, kInputPatchSize))
            newForgeries = generatorModel.predict(tf.constant(inputs, 'float32'))
            forgeries = [*oldForgeries, *newForgeries]
            inputs = []
            outputs = []
            for count in range(100):
                if random.random()>0.5 :
                    pop, forgeries = forgeries[-1], forgeries[:-1]
                    inputs.append(pop)
                    outputs.append(0)
                else:
                    inputs.append(getPatch(painting))
                    outputs.append(1)
            discriminatorModel.trainable=True
            print("Train discriminator from output of generator.")
            #if the model has gotten lost, reset the weights
            if epochs > 10:
                discriminatorModel = getDiscriminatorModel()
            discriminatorAccuracy = trainModel(discriminatorModel,inputs,outputs,name="retrained_discriminator",epochs=epochs)
            print("Discriminator Acuracy = " + str(discriminatorAccuracy))
            oldForgeries = forgeries.copy()
            np.random.shuffle(oldForgeries)
            oldForgeries = oldForgeries[:int(len(oldForgeries)/2)]
            epochs = epochs + 1

if __name__ == "__main__":
    main()
