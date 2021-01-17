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
#tImage = cv2.cvtColor(tImage, cv2.COLOR_BGR2GRAY)
#truth = cv2.resize(tImage,(256,256))
#cv2.imwrite("C:\\Temp\\truth.png", truth)
tImage = cv2.resize(truth,(64,64))
imageArray = []
imageArray.append(np.asarray(tImage)/ 255.)
#bicubic = cv2.resize(tImage,(256,256))
#cv2.imwrite("C:\\Temp\\input.png", bicubic)

f = open("psnrLog.txt", "w")
f.writelines("PSNR before: (100 is a perfect score)\n")
f.writelines(str(PSNR(cv2.resize(truth,(256,256)),cv2.resize(tImage,(256,256))))+"\n")


f.writelines("PSNR after:\n")
for i in range(1,5001,100):
    model = load_model("check_%d.hdf5"%i, custom_objects={'my_psnr': my_psnr})
    output = model.predict(np.asarray(imageArray))
    cv2.imwrite("C:\\Temp\\output_epoc_%d.png"%i, output[0]*255)
    f.writelines(str(PSNR(cv2.resize(truth,(256,256)),cv2.resize(output[0]*255,(256,256))))+"\n")
    f.flush()
f.close()
    