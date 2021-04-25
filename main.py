import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from keras.preprocessing import image
import cv2
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

training_set = train_datagen.flow_from_directory('C:\\python cnn\\python cnn\\train',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('C:\\python cnn\\python cnn\\test',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

print("Image Processing.... Completed")

#TRAINING OUT MODE (CNN MODEL)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#COMPILE THE MODEL
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#FIT THE MODEL
model.fit(x = training_set, validation_data = test_set, epochs=25)

vid = cv2.VideoCapture(1)
print("Camera connection Successfully established ")
i = 0
while(True):
    r, image = vid.read()
    cv2.imshow('image', image)
    cv2.imwrite('C:\\python cnn\\python cnn\\final'+str(i)+".jpg", image)
    test_image = image.load_img('C:\\python cnn\\python cnn\\final'+str(i)+".jpg", target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1:
        print("This is A Banana , Hii ni Ndizi")
    if result[0][0] == 0:
        print("This is An Apple")
    os.remove('C:\\python cnn\\python cnn\\final'+str(i)+".jpg")
    i=1+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()


