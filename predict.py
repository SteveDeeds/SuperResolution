import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import glob
import os
import numpy as np
import train_matt
import bigger_model as model_def


kWriteInputPatches = False


def imageFromPatches(patches, verticalPatchCount):
    resultImage = np.empty(
        (0, model_def.kOutputPatchSize * verticalPatchCount, 3), dtype='float32')

    i = 0
    while (i < len(patches)):
        resultLine = np.empty(
            (model_def.kOutputPatchSize, 0, 3), dtype='float32')
        for y in range(0, verticalPatchCount):
            resultLine = np.append(resultLine, patches[i], axis=1)
            i += 1
        resultImage = np.append(resultImage, resultLine, axis=0)

    return resultImage


def main():
    fileNames = glob.glob(os.path.join("input", "*.jpg"))
    fileNames.extend(glob.glob(os.path.join("input", "*.png")))
    img = cv2.imread(fileNames[0]) / 255.0
    img = cv2.resize(img,(256,256))
    cv2.imwrite(os.path.join("eval", "truth.png"), img * 255.0)
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
    model = load_model("checkpoint.hdf5",
                       custom_objects={'my_psnr': model_def.my_psnr})
    results = model.predict(patches)
    # # output = imageFromPatches(results)
    # i = 0
    # for patch in results:
    #     cv2.imwrite(
    #         os.path.join("eval", "output-patch" + str(i) + ".png"),
    #         patch * 255.0)
    #     i = i + 1
    verticalPatchCount = int(
        (img.shape[1]-model_def.kInputPadding*2)  # Should be img.shape[1]
        / model_def.kOutputPatchSize)

    # Because model loading failed, I'm using 'patches' here but
    # it really should be 'results'.
    assembled = imageFromPatches(results, verticalPatchCount)
    print("Final shape: " + str(assembled.shape))
    cv2.imwrite(os.path.join("eval", "assembled.png"), assembled * 255.0)


if __name__ == "__main__":
    main()
