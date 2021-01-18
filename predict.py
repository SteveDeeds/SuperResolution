import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from keras.models import load_model
from math import log10, sqrt
from tensorflow.keras import backend as K

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def my_psnr(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return -100.0
    max_pixel = 1.0
    psnr = 20 * K.log(max_pixel / K.sqrt(mse))
    return -psnr

truth = cv2.imread("C:\\ffhq\\00000.png")
#truth = cv2.imread("C:\\ffhq\\hide\\00010.png")
# tImage = cv2.resize(truth,(65,65), interpolation=cv2.INTER_CUBIC)
# truth = cv2.resize(truth,(260,260))
# truth = truth[2:258, 2:258]

tImage = cv2.resize(truth,(270,270))
truth = tImage[7:263, 7:263]
tImage = cv2.resize(tImage,(64,64), interpolation=cv2.INTER_CUBIC)
tImage = cv2.resize(tImage,(270,270), interpolation=cv2.INTER_CUBIC)

imageArray = []
imageArray.append(np.asarray(tImage)/ 255.)
cv2.imwrite("C:\\Temp\\input.png", tImage[7:263, 7:263])
cv2.imwrite("C:\\Temp\\truth.png", truth)

f = open("psnrLog.txt", "w")
f.writelines("PSNR before: (100 is a perfect score)\n")
f.writelines(str(PSNR(truth,cv2.resize(tImage,(256,256))))+"\n")


f.writelines("PSNR after:\n")
for i in range(100,5001,100):
    model = load_model("check_%d.hdf5"%i, custom_objects={'my_psnr': my_psnr})
    output = model.predict(np.asarray(imageArray))
    cv2.imwrite("C:\\Temp\\output_epoc_%d.png"%i, output[0]*255)
    f.writelines(str(PSNR(truth,output[0]))+"\n")
    f.flush()
f.close()
