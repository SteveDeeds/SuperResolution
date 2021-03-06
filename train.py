import tensorflow as tf
import numpy as np
import cv2
import glob
from tensorflow.keras import backend as K

sharpArray = []
bluryArray = []

def noisy(noise_typ,image):
  if noise_typ == "gauss":
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy
  elif noise_typ == "s&p":
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1
    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    return out
  # elif noise_typ == "poisson":
  #     vals = len(np.unique(image))
  #     vals = 2 ** np.ceil(np.log2(vals))
  #     noisy = np.random.poisson(image * vals) / float(vals)
  #     return noisy
  # elif noise_typ =="speckle":
  #     row,col,ch = image.shape
  #     gauss = np.random.randn(row,col,ch)
  #     gauss = gauss.reshape(row,col,ch)        
  #     noisy = image + image * gauss
  #     return noisy

def loadImages():
    fileNames = glob.glob("C:\\ffhq\\*.png")
    for fname in fileNames:
        tImage = cv2.imread(fname)
        tImage = cv2.resize(tImage,(270,270))
        #tImage = cv2.resize(tImage,(260,260))
        sharpImage = tImage[7:263, 7:263]
        #sharpImage = tImage[2:258, 2:258]
        sharpArray.append(np.asarray(sharpImage) / 255.)
        tImage = cv2.resize(tImage,(64,64), interpolation=cv2.INTER_CUBIC)
        tImage = cv2.resize(tImage,(270,270), interpolation=cv2.INTER_CUBIC)
        #tImage = cv2.resize(tImage,(65,65), interpolation=cv2.INTER_CUBIC)
        #tImage = desaturate(tImage, 0.5)
        #tImage = tImage / 10. # darken
        #row,col = tImage.shape
        #gauss = np.random.randn(row,col)
        #gauss = gauss.reshape(row,col)        
        #tImage = tImage + tImage * gauss
        #bluryArray.append(np.expand_dims(np.asarray(tImage),axis=2)/ 255.)
        bluryArray.append(np.asarray(tImage)/ 255.)
    cv2.imwrite("C:\\Temp\\bicubic.png", cv2.resize(bluryArray[0]*255,(256,256)))
    cv2.imwrite("C:\\Temp\\truth.png", sharpArray[0]*255)

def desaturate(img,satadj):
  imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
  (h, s, v) = cv2.split(imghsv)
  s = s*satadj
  s = np.clip(s,0,255)
  imghsv = cv2.merge([h,s,v])
  imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
  return imgrgb


class CenterAround(tf.keras.constraints.Constraint):
  """Constrains weight tensors to be centered around `ref_value`."""

  def __init__(self, ref_value):
    self.ref_value = ref_value

  def __call__(self, w):
    mean = tf.reduce_mean(w)
    return w - mean + self.ref_value

  def get_config(self):
    return {'ref_value': self.ref_value}

def my_psnr(y_true, y_pred):
  temp_true = y_true * 255
  temp_pred = y_pred * 255
  mse = K.mean(K.square(temp_true - temp_pred)) 
  if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
      return -100.0
  max_pixel = 255
  psnr = 20 * K.log(max_pixel / K.sqrt(mse)) 
  return -psnr 
  # #difference between true label and predicted label
  # error = y_true-y_pred    
  # #square of the error
  # sqr_error = K.square(error)
  # #mean of the square of the error
  # mean_sqr_error = K.mean(sqr_error)
  # #square root of the mean of the square of the error
  # sqrt_mean_sqr_error = K.sqrt(mean_sqr_error)
  # #return the error
  # return sqrt_mean_sqr_error

loadImages()

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input(shape=(65,65,3)))
# model.add(tf.keras.layers.Conv2DTranspose(256, 3, strides=2, output_padding=1, activation='linear', padding='same'))
# model.add(tf.keras.layers.Conv2DTranspose(128, 3, strides=2, output_padding=1, activation='linear', padding='same'))
# model.add(tf.keras.layers.Conv2D(3, 5, activation='hard_sigmoid', padding='valid'))
# model.compile(loss=my_psnr, optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mean_squared_error','accuracy'])
# model.summary()

# l_input = tf.keras.layers.Input(shape=(270,270,3))
# l_input_crop = tf.keras.layers.Cropping2D(7)(l_input)
# #l_h1 = tf.keras.layers.Conv2DTranspose(3, 3, strides=4, output_padding=3, activation='linear', padding='same')(l_input)
# l_h2 = tf.keras.layers.Conv2D(256, 9, activation='relu', padding='valid')(l_input)
# l_h3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='valid')(l_h2)
# l_h4 = tf.keras.layers.Conv2D(3, 5, activation='hard_sigmoid', padding='valid')(l_h3)
# l_output = tf.keras.layers.Add()([l_input_crop, l_h4])
# model = tf.keras.models.Model(inputs=l_input, outputs=l_output)
# model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mean_squared_error','accuracy'])
# model.summary()


# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input(shape=(270,270,3)))
# model.add(tf.keras.layers.Conv2D(128, 9, activation='relu', padding='valid'))
# model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid'))
# model.add(tf.keras.layers.Conv2D(3, 5, activation='hard_sigmoid', padding='valid'))
# model.compile(loss=my_psnr, optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mean_squared_error','accuracy'])
# model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint("check_5k+{epoch}.hdf5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False)

from keras.models import load_model
model = load_model("check_5000.hdf5", custom_objects={'my_psnr': my_psnr})

history = model.fit(x=np.asarray(bluryArray), y=np.asarray(sharpArray), epochs=50000, batch_size=10,callbacks=[checkpoint],verbose=1)
print(history.history['loss'])
print("Finished training the model")

output = model.predict(np.asarray(bluryArray[0:1]))

cv2.imwrite("C:\\Temp\\output.png", output[0]*255)
print("saved output")