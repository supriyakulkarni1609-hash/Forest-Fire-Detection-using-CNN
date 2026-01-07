import os
import math
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
"""
This module contains functions used to create training and validation datasets using with proper representation of each
class. It also contains a batch generator which performs data augmentation (shifts, rotations, flips and zooms) on the
fly. Finally, transfer learning from an InceptionV3-based model is performed and the model is re-trained for fire
images using our augmented dataset.
"""

# we work with three classes for this whole project
classes = ['fire', 'no_fire', 'start_fire']
nbr_classes = 3

def augmented_batch_generator(images_paths, labels, batch_size, preprocessing, augment, image_size=(224, 224)):
    display = False
    number_samples = len(images_paths)

    if augment:
        data_transformer = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            rotation_range=20,
            brightness_range=[0.7, 1.3],
            zoom_range=[0.8, 1.3]
        )

    while True:
        perm = np.random.permutation(number_samples)
        images_paths = images_paths[perm]
        labels = labels[perm]

        for i in range(0, number_samples, batch_size):
            batch = list(map(
                lambda x: image.load_img(x, target_size=image_size),
                images_paths[i:i + batch_size]
            ))

            if augment:
                batch = np.array(list(map(
                    lambda x: data_transformer.random_transform(image.img_to_array(x)),
                    batch
                )))
            else:
                batch = np.array(list(map(
                    lambda x: image.img_to_array(x),
                    batch
                )))

            if display:
                for j in range(9):
                    plt.subplot(330 + 1 + j)
                    plt.imshow(batch[j].astype('uint8'))
                    print(labels[j])

            batch = preprocessing(batch)
            yield batch, labels[i:i + batch_size]


def extract_dataset(dataset_path, classes_names, percentage):
    num_classes = len(classes_names)

    def listdir_nohidden(path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f

    train_labels, val_labels = np.empty([1, 0]), np.empty([1, 0])
    train_samples, val_samples = np.empty([1, 0]), np.empty([1, 0])

    for class_name in listdir_nohidden(dataset_path):
        images_paths, labels = [], []
        class_path = os.path.join(dataset_path, class_name)
        class_id = classes_names.index(class_name)

        for file in listdir_nohidden(class_path):
            path = os.path.join(class_path, file)
            try:
                Image.open(path).verify()
            except:
                continue
            images_paths.append(path)
            labels.append(class_id)

        labels = np.array(labels)
        images_paths = np.array(images_paths)

        perm = np.random.permutation(len(images_paths))
        labels = labels[perm]
        images_paths = images_paths[perm]

        border = math.floor(percentage * len(images_paths))

        train_labels = np.append(train_labels, labels[:border])
        val_labels = np.append(val_labels, labels[border:])
        train_samples = np.append(train_samples, images_paths[:border])
        val_samples = np.append(val_samples, images_paths[border:])

    perm = np.random.permutation(len(train_samples))
    train_labels = to_categorical(train_labels, num_classes)[perm]
    train_samples = train_samples[perm]

    perm = np.random.permutation(len(val_samples))
    val_labels = to_categorical(val_labels, num_classes)[perm]
    val_samples = val_samples[perm]

    print(f"Training on {len(train_samples)} samples")
    print(f"Validation on {len(val_samples)} samples")

    return (train_samples, train_labels), (val_samples, val_labels)


def create_inception_based_model():
    base_model = InceptionV3(
        include_top=False,
        weights='imagenet',
        pooling='max',
        input_shape=(224, 224, 3)
    )

    x = base_model.output
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(nbr_classes, activation='softmax')(x)

    model = Model(inputs=base_model.inputs, outputs=predictions)

    for layer in model.layers:
        layer.trainable = True

    return model


def train_inception_based_model(dataset_path,
                                fine_tune_existing=None,
                                learning_rate=0.001,
                                percentage=0.9,
                                nbr_epochs=3,
                                batch_size=32):

    if fine_tune_existing:
        model = load_model(fine_tune_existing)
    else:
        model = create_inception_based_model()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(
        dataset_path, classes, percentage
    )

    train_gen = augmented_batch_generator(
        train_samples, train_labels, batch_size,
        inception_preprocess_input, augment=True
    )

    val_gen = augmented_batch_generator(
        val_samples, val_labels, batch_size,
        inception_preprocess_input, augment=False
    )

    model.fit(
        train_gen,
        steps_per_epoch=math.ceil(len(train_samples) / batch_size),
        epochs=nbr_epochs,
        validation_data=val_gen,
        validation_steps=math.ceil(len(val_samples) / batch_size),
        verbose=1
    )
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(BASE_DIR, "final_fire_model.h5")

    model.save(model_save_path)
    print("Model saved at:", model_save_path)

if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(BASE_DIR, "dataset")

    print("Running from:", BASE_DIR)
    print("Looking for dataset at:", dataset_path)

    train_inception_based_model(
        dataset_path=dataset_path,
        learning_rate=0.001,
        percentage=0.9,
        nbr_epochs=10,
        batch_size=32
    )
