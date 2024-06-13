from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import ZeroPadding2D
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
import splitfolders
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.python.client import device_lib

TRAINING_LOGS_FILE = "training_logs.csv"
MODEL_SUMMARY_FILE = "model_summary.txt"
MODEL_FILE = "model_test.hdf5"

def get_directory_list(path):
    import os
    directories = []
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            directories.append(name)
    return directories


def split_folder(src_folder, dest_folder, train_ratio, test_ratio, validation_ratio):
    splitfolders.ratio(src_folder, output=dest_folder,
                       seed=1337, ratio=(train_ratio, test_ratio, validation_ratio), group_prefix=None,
                       move=False)  # default values
    training_data_dir = dest_folder + "train"
    test_data_dir = dest_folder + "test"
    validation_data_dir = dest_folder + "val"

    return training_data_dir, test_data_dir, validation_data_dir


def preprocessing_data(training_data_dir, test_data_dir, validation_data_dir):
    train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                              shear_range=0.2,
                                              zoom_range=0.2,
                                              horizontal_flip=True)
    test_data_generator = ImageDataGenerator(rescale=1. / 255)
    validation_data_generator = ImageDataGenerator(rescale=1. / 255)

    set_train_data = train_data_generator.flow_from_directory(training_data_dir,
                                                              target_size=(64, 64),
                                                              batch_size=23,
                                                              class_mode='categorical')
    set_test_data = test_data_generator.flow_from_directory(test_data_dir,
                                                            target_size=(64, 64),
                                                            batch_size=23,
                                                            class_mode='categorical')
    set_val_data = validation_data_generator.flow_from_directory(validation_data_dir,
                                                                 target_size=(64, 64),
                                                                 batch_size=23,
                                                                 class_mode='categorical')

    return set_train_data, set_test_data, set_val_data


def create_model():
    # Initialising the CNN
    cnn_model = Sequential()

    # Con-1
    cnn_model.add(Conv2D(32, (2, 2), input_shape=(64, 64, 3), activation='relu'))
    cnn_model.add(BatchNormalization())

    cnn_model.add(Conv2D(32, (2, 1), activation='relu'))
    cnn_model.add(BatchNormalization())

    cnn_model.add(Conv2D(32, (1, 2), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    # COn-2
    cnn_model.add(Conv2D(48, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())

    cnn_model.add(Conv2D(48, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())

    cnn_model.add(Conv2D(48, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    # Con-3
    cnn_model.add(Conv2D(64, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())

    cnn_model.add(Conv2D(64, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())

    cnn_model.add(Conv2D(64, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    # COn-4
    cnn_model.add(Conv2D(80, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())

    cnn_model.add(Conv2D(80, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())

    cnn_model.add(Conv2D(80, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    # Con-5
    cnn_model.add(ZeroPadding2D((1, 1)))
    cnn_model.add(Conv2D(96, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())

    cnn_model.add(ZeroPadding2D((1, 1)))
    cnn_model.add(Conv2D(96, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())

    cnn_model.add(ZeroPadding2D((1, 1)))
    cnn_model.add(Conv2D(96, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    # Con-6
    cnn_model.add(ZeroPadding2D((1, 1)))
    cnn_model.add(Conv2D(112, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())

    cnn_model.add(ZeroPadding2D((1, 1)))
    cnn_model.add(Conv2D(112, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())

    cnn_model.add(ZeroPadding2D((1, 1)))
    cnn_model.add(Conv2D(112, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten
    cnn_model.add(Flatten())

    # CLassifier (Fully Conect.)
    cnn_model.add(Dense(units=800, activation='relu', kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(0.00001)))
    cnn_model.add(Dense(units=800, activation='relu', kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(0.00001)))
    cnn_model.add(Dense(units=400, activation='relu', kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(0.00001)))
    cnn_model.add(Dense(units=200, activation='relu', kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(0.00001)))
    cnn_model.add(Dense(units=3, activation='softmax'))

    return cnn_model


def show_plot(history):
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


training_data_dir, test_data_dir, validation_data_dir = split_folder("Datasets", "Datasets/images_split/", 0.75, 0.2,
                                                                     0.05)

set_train_data, set_test_data, set_val_data = preprocessing_data(training_data_dir, test_data_dir, validation_data_dir)

classifier = create_model()
from keras.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

Adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00014, amsgrad=False)
classifier.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])

with open(MODEL_SUMMARY_FILE, "w") as fh:
    classifier.summary(print_fn=lambda line: fh.write(line + "\n"))

csv_logger = CSVLogger('training_logs.csv')
history = classifier.fit_generator(set_train_data,
                                   epochs=50,
                                   validation_data=set_test_data,
                                   verbose=1,
                                   callbacks=[checkpoint_callback])
show_plot(history)
