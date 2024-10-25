import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Activation, Add, Multiply, Flatten, Dense, Input
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Attention Block Function
def attention_block(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, (1, 1), strides=(1, 1))(x)
    phi_g = Conv2D(inter_channel, (1, 1), strides=(1, 1))(g)

    add_xg = Add()([theta_x, phi_g])
    relu_xg = Activation('relu')(add_xg)

    psi = Conv2D(1, (1, 1), strides=(1, 1))(relu_xg)
    sigmoid_xg = Activation('sigmoid')(psi)

    upsample_psi = UpSampling2D(size=(x.shape[1] // psi.shape[1], x.shape[2] // psi.shape[2]))(sigmoid_xg)
    multiply_xg = Multiply()([upsample_psi, x])

    return multiply_xg

# Attention U-Net for Classification
def attention_unet_classification(input_size=(256, 256, 3), num_classes=3):
    inputs = Input(input_size)

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    # Decoder with Attention
    up6 = Conv2D(512, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    att6 = attention_block(conv4, up6, 512)
    merge6 = concatenate([att6, up6], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv2D(256, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    att7 = attention_block(conv3, up7, 256)
    merge7 = concatenate([att7, up7], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    att8 = attention_block(conv2, up8, 128)
    merge8 = concatenate([att8, up8], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    att9 = attention_block(conv1, up9, 64)
    merge9 = concatenate([att9, up9], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    # Output Layer for Classification (num_classes)
    flatten = Flatten()(conv9)
    dense1 = Dense(128, activation='relu')(flatten)
    output = Dense(num_classes, activation='softmax')(dense1)

    model = Model(inputs, output)

    return model

# Load Dataset
def load_images_from_folder(folder, img_size=(256, 256)):
    images = []
    labels = []
    class_names = ['benign', 'malignant', 'normal']

    for class_folder in os.listdir(folder):
        img_path = os.path.join(folder, class_folder)
        class_index = class_names.index(class_folder)  # Assign label based on folder name

        for img_file in os.listdir(img_path):
            img = cv2.imread(os.path.join(img_path, img_file))
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(class_index)

    return np.array(images), np.array(labels)

# Example: Load dataset (update path as per your setup)
X, y = load_images_from_folder(r'C:\Users\Pratiksha\Downloads\Dataset_BUSI\Dataset_BUSI_with_GT', img_size=(128, 128))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Compile model
model = attention_unet_classification(input_size=(128, 128, 3), num_classes=3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_split=0.1, epochs=30, batch_size=16)

# Save the trained model
model.save('attention_unet_classification_model.keras')

# Evaluate model
model.evaluate(X_test, y_test)


# Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Load saved model for testing
def load_and_predict(input_image_path):
    # Load the model
    saved_model = load_model('attention_unet_classification_model.keras', compile=False)

    # Load and preprocess the input image
    img = cv2.imread(input_image_path)
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict the class
    prediction = saved_model.predict(img_input)
    class_names = ['benign', 'malignant', 'normal']
    predicted_class = class_names[np.argmax(prediction)]

    # Display the input image and predicted class
    plt.imshow(img_resized)
    plt.title(f'Predicted: {predicted_class}')
    plt.show()

# Example of predicting a new image
load_and_predict(r'C:\Users\Pratiksha\Downloads\Dataset_BUSI\Dataset_BUSI_with_GT/malignant/malignant (100).png')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot to class labels
y_true_classes = np.argmax(y_test, axis=1)  # Convert one-hot to class labels

# Compute the confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['benign', 'malignant', 'normal'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
# Load saved model for testing
def load_and_predict(input_image_path):
    # Load the model
    saved_model = load_model('attention_unet_classification_model.keras', compile=False)

    # Load and preprocess the input image
    img = cv2.imread(input_image_path)
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict the class
    prediction = saved_model.predict(img_input)
    class_names = ['benign', 'malignant', 'normal']
    predicted_class = class_names[np.argmax(prediction)]

    # Display the input image and predicted class
    plt.imshow(img_resized)
    plt.title(f'Predicted: {predicted_class}')
    plt.show()

# Example of predicting a new image
load_and_predict(r'C:\Users\Pratiksha\Downloads\Dataset_BUSI\Dataset_BUSI_with_GT/benign/benign (121).png')
# Load saved model for testing
def load_and_predict(input_image_path):
    # Load the model
    saved_model = load_model('attention_unet_classification_model.keras', compile=False)

    # Load and preprocess the input image
    img = cv2.imread(input_image_path)
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict the class
    prediction = saved_model.predict(img_input)
    class_names = ['benign', 'malignant', 'normal']
    predicted_class = class_names[np.argmax(prediction)]

    # Display the input image and predicted class
    plt.imshow(img_resized)
    plt.title(f'Predicted: {predicted_class}')
    plt.show()

# Example of predicting a new image
load_and_predict(r'C:\Users\Pratiksha\Downloads\Dataset_BUSI\Dataset_BUSI_with_GT/malignant/malignant (100).png')
# Load saved model for testing
def load_and_predict(input_image_path):
    # Load the model
    saved_model = load_model('attention_unet_classification_model.keras', compile=False)

    # Load and preprocess the input image
    img = cv2.imread(input_image_path)
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict the class
    prediction = saved_model.predict(img_input)
    class_names = ['benign', 'malignant', 'normal']
    predicted_class = class_names[np.argmax(prediction)]

    # Display the input image and predicted class
    plt.imshow(img_resized)
    plt.title(f'Predicted: {predicted_class}')
    plt.show()

# Example of predicting a new image
load_and_predict(r'C:\Users\Pratiksha\Downloads\Dataset_BUSI\Dataset_BUSI_with_GT/normal/normal (10).png')
