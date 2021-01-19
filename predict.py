import tensorflow as tf
from keras.models import load_model
import cv2
import glob
import os
import numpy as np
import train_matt
import bigger_model as model_def

def imageFromPatches(patches):
    # take a list of patches and make one image out of them by tileing them.
    return output

def main():
    fileNames = glob.glob(os.path.join("input", "*.jpg"))
    fileNames.extend(glob.glob(os.path.join("input", "*.png")))
    img = cv2.imread(fileNames[0])
    img = cv2.resize(img,(256,256))
    img = cv2.resize(img,(1024,1024))
    img = img / 255.0
    patches = train_matt.getCoveringPatches(img,model_def.kInputPatchSize,model_def.kInputPadding)
    i=0
    for patch in patches:
        cv2.imwrite("input"+str(i)+".png", patch*255.0)
        i=i+1
    model = load_model("checkpoint.hdf5", custom_objects={'my_psnr': model_def.my_psnr})
    results = model.predict(patches)
    #output = imageFromPatches(results)
    i=0
    for patch in results:
        cv2.imwrite("output"+str(i)+".png", patch*255.0)
        i=i+1

if __name__ == "__main__":
    main()