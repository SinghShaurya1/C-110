import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("keras_model.h5")

vid = cv2.VideoCapture(0)
  
while(True):
    ret, frame = vid.read()
  
    img = cv2.resize(frame, (224, 224))

    testImg = np.array(img, dtype=np.float32)

    testImg = np.expand_dims(testImg, axis=0)

    normalized = testImg/255.0

    predictions =  model.predict(normalized)
    print('predicted vals:', predictions)


    cv2.imshow('frame', frame)
      
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
vid.release()

cv2.destroyAllWindows()