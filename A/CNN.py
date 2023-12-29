from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt


def create_and_compile_model(input_shape=(28, 28, 1), dropout_rates=(0.1, 0.2), dense_units=128, output_units=1, activation='sigmoid', optimizer='rmsprop', loss='binary_crossentropy'):
    # Create a Sequential model
    model = Sequential()

    # Convolutional layers with BatchNormalization and MaxPooling
    model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))

    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(dropout_rates[0]))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))

    model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(dropout_rates[1]))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))

    # Flatten and Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dropout(dropout_rates[1]))
    model.add(Dense(units=output_units, activation=activation))

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


def apply_data_augmentation(x_train, y_train, batch_size=32, rotation_range=30, zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=False):
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip
    )

    augmented_data = datagen.flow(x_train, y_train, batch_size=batch_size)
    return augmented_data


def train_model_with_augmentation(model, augmented_data, x_val, y_val, epochs=12):
    # Learning rate reduction callback
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

    # Training the model with data augmentation and learning rate reduction
    history = model.fit(augmented_data, epochs=epochs, validation_data=(x_val, y_val), callbacks=[learning_rate_reduction])
    return history


def plot_training_history(history):
    epochs = range(1, len(history.history['accuracy']) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Training and Validation Accuracy
    ax[0].plot(epochs, history.history['accuracy'], 'o-', label='Training Accuracy', color='#9f1f31')
    ax[0].plot(epochs, history.history['val_accuracy'], 'o-', label='Validation Accuracy', color='#03608C')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].grid(False)

    # Plot Training and Validation Loss
    ax[1].plot(epochs, history.history['loss'], 'o-', label='Training Loss', color='#9f1f31')
    ax[1].plot(epochs, history.history['val_loss'], 'o-', label='Validation Loss', color='#03608C')
    ax[1].set_title('Training & Validation Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].grid(False)

    plt.show()