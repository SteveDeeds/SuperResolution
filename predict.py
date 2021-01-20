import tensorflow as tf
from keras.models import load_model
import cv2
import glob
import os
import numpy as np
import train_matt
import bigger_model as model_def


kWriteInputPatches = False


def imageFromPatches(patches, horizontalPatchCount):
    resultImage = np.empty((0, 0, model_def.kInputPatchSize, 3), dtype='float')

    i = 0
    while (i < len(patches)):
        resultLine = np.empty(
            (0, 0, model_def.kInputPatchSize, 3), dtype='float')
        for x in range(0, horizontalPatchCount):
            resultLine = np.append(resultImage, patches[i], axis=1)
        resultImage = np.append(resultImage, resultLine, axis=0)

        return resultImage


def main():
    fileNames = glob.glob(os.path.join("input", "*.jpg"))
    fileNames.extend(glob.glob(os.path.join("input", "*.png")))
    img = cv2.imread(fileNames[0]) / 255.0
    blurry = train_matt.getBlurryImage(img)
    patches = train_matt.getCoveringPatches(
        blurry, model_def.kInputPatchSize, model_def.kInputPadding)
    if (kWriteInputPatches):
        i = 0
        for patch in patches:
            cv2.imwrite(
                os.path.join("eval", "input-patch" + str(i) + ".png"),
                patch * 255.0)
            i = i + 1
    # Loading the model fails for me, so I'm commenting this out
    # model = load_model("check_10.hdf5",
    #                    custom_objects={'my_psnr': model_def.my_psnr})
    # results = model.predict(patches)
    # # output = imageFromPatches(results)
    # i = 0
    # for patch in results:
    #     cv2.imwrite(
    #         os.path.join("eval", "output-patch" + str(i) + ".png"),
    #         patch * 255.0)
    #     i = i + 1
    horizontalPatchCount = (
        blurry.shape[0]
        / (model_def.kInputPatchSize - 2 * model_def.kInputPadding))

    # Because model loading failed, I'm using 'patches' here but
    # it really should be 'results'.
    assembled = imageFromPatches(patches, horizontalPatchCount)
    cv2.imwrite(assembled * 255, os.path.join("eval", "assembled.png"))


if __name__ == "__main__":
    main()
