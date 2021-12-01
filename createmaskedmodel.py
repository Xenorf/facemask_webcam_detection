import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

# Images information
IMG_WIDTH = 128
IMG_HEIGHT = 128
img_folder = 'masked_database/'

# Creating the dataset to train the model
def create_dataset(img_folder):
    img_data_array = []
    class_name = []

# Iterating over all the pictures in the dataset
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1,  file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            # Resizing and normalizing the images to use them to train the model
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),
                               interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            # Associating the image with the class it belongs to
            class_name.append(dir1)
    return img_data_array, class_name

img_data, class_name = create_dataset(img_folder)

# Converting classes (yes,no) into numbers (0,1)
target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
target_val = [target_dict[class_name[i]] for i in range(len(class_name))]
targets = np.array(list(map(int, target_val)), np.float32)
images = np.array(img_data, np.float32)

# Keeping 20% of the data to test the model to detect the overfitting of the model
images_train, images_test, targets_train, targets_test = train_test_split(
    images, targets, test_size=0.2, random_state=1)

# Model creation
model = tf.keras.models.Sequential()
# layer 1 : Convolution with 3*3 filters
model.add(tf.keras.layers.Conv2D(100, (3, 3), activation='relu',
          input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
# layer 2 : Pooling, images get a 2*2 filter applied and only the highest value is kept
model.add(tf.keras.layers.MaxPooling2D(2, 2))
# layer 3 : Convert image matrix into a vector
model.add(tf.keras.layers.Flatten())
# layer 4 : Randomly disabling half of the neurons to minimize overfitting
model.add(tf.keras.layers.Dropout(0.5))
# layer 5 : Unlinear activation
model.add(tf.keras.layers.Dense(50, activation="relu"))
# layer 6 : Probability distribution, between 0 and 1
model.add(tf.keras.layers.Dense(2, activation="softmax"))

# Model compilation
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["sparse_categorical_accuracy"]
)

# Model training
history = model.fit(images_train, targets_train,
                    epochs=10, validation_split=0.2)

# Model testing
loss, acc = model.evaluate(images_test, targets_test)
print("Test Loss", loss)
print("Test Accuracy", acc)

# Get the performance statistics of the model
loss_curve_train = history.history["loss"]
acc_curve_train = history.history["sparse_categorical_accuracy"]
acc_curve_validation = history.history["val_sparse_categorical_accuracy"]
loss_curve_validation = history.history["val_loss"]

# Saving the trained model in a file
model.save("masked.h5")

# Print of the loss accuracy of training and testing data
plt.figure(1)
plt.plot(acc_curve_train)
plt.plot(acc_curve_validation, color="red")
plt.title("Accuracy")

# Print of the loss function of training and testing data
plt.figure(2)
plt.plot(loss_curve_train)
plt.plot(loss_curve_validation, color="red")
plt.title("Loss")
plt.show()
