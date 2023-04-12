import os  # interacting with operating system
import cv2  # to load and process images
import numpy as np  # to work with numpy arrays
import matplotlib.pyplot as plt  # visualization of actual digits
import tensorflow as tf  # for implementing the ML part


mnist = tf.keras.datasets.mnist #dataset from tensorflow module
(x_train, y_train), (x_test, y_test) = mnist.load_data() #divide into 2 tuples training data and test data
#x is pixel data (image) and y is classification (number)

#Normalize the pixel data between 0 to 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)



#preparing the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

print("Training the Model:")
#compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'] )
#training the model
model.fit(x_train, y_train, epochs=3)
#save the model
model.save('handwritten.model')


# loading the saved model
model = tf.keras.models.load_model('handwritten.model')

print("Testing the Model on Prelabelled data:")
loss, accuracy = model.evaluate(x_test, y_test)
#print the loss and accuracy of the model on tf data
print(loss) #as low as possible
print(accuracy) #as high as possible

print("Implementing the Model on Hand-written data:")
#Implementing the model on self_written digits
image_number = 0
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1
