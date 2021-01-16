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
        tImage = cv2.resize(tImage,(256,256))
        #tImage = cv2.cvtColor(tImage, cv2.COLOR_BGR2GRAY)
        sharpArray.append(np.asarray(tImage) / 255.)
        tImage = cv2.resize(tImage,(64,64))
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
    mse = K.mean(K.square(y_true - y_pred)) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100.0
    max_pixel = 255.0
    psnr = 20 * K.log(max_pixel / K.sqrt(mse)) 
    return psnr 
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

l_input = tf.keras.layers.Input(shape=(64,64,3))
l_scaled_input = tf.keras.layers.UpSampling2D((4,4), interpolation="bilinear")(l_input)
l_h1 = tf.keras.layers.ZeroPadding2D(5)(l_scaled_input)
l_h2 = tf.keras.layers.Conv2D(256, 5, activation='relu', padding='valid')(l_h1)
l_h3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='valid')(l_h2)
#l_h4 = tf.keras.layers.UpSampling2D((4,4), interpolation="bilinear")(l_h3)
#l_h5 = tf.keras.layers.ZeroPadding2D(2)(l_h4)
l_h6 = tf.keras.layers.Conv2D(3, 5, activation='linear', padding='valid')(l_h3)
l_output = tf.keras.layers.Add()([l_scaled_input, l_h6])
model = tf.keras.models.Model(inputs=l_input, outputs=l_output)
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mean_squared_error','accuracy'])


# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input(shape=(64,64,3)))
# model.add(tf.keras.layers.UpSampling2D((4,4), interpolation="bilinear"))
# model.add(tf.keras.layers.Conv2D(128, 9, activation='relu', padding='same'))
# model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
# model.add(tf.keras.layers.Conv2D(3, 5, activation='linear', padding='same', kernel_initializer=tf.keras.initializers.Zeros()))
# model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mean_squared_error','accuracy'])
#model.compile(loss=my_psnr, optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mean_squared_error','accuracy'])
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint("check_{epoch}.hdf5", monitor='val_loss', verbose=1, save_best_only=False,
                                 save_weights_only=False)


history = model.fit(x=np.asarray(bluryArray), y=np.asarray(sharpArray), epochs=5000, batch_size=10,callbacks=[checkpoint],verbose=1)
print(history.history['loss'])
print("Finished training the model")

output = model.predict(np.asarray(bluryArray[0:1]))

cv2.imwrite("C:\\Temp\\output.png", output[0]*255)
print("saved output")