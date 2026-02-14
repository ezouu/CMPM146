"""
Preprocess image data for training: read images, RGB grids, float tensors.
Rescaling to [0, 1] is done in the model (e.g. Rescaling(1./255) in BasicModel).

1. Read in image files          -> image_dataset_from_directory(..., directory)
2. Preprocess JPEG to RGB        -> color_mode='rgb'
3. Float tensors                 -> dataset yields float32 batches
4. Rescale to 0-1                -> model layer Rescaling(1./255)
"""
from tensorflow.keras.utils import image_dataset_from_directory
from config import train_directory, test_directory, image_size, batch_size, validation_split

def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='both',
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    return train_dataset, validation_dataset, test_dataset

def get_transfer_datasets():
    # Your code replaces this by loading the dataset
    # you can use image_dataset_from_directory, similar to how the _split_data function is using it
    train_dataset, validation_dataset, test_dataset = None, None, None
    # ...
    return train_dataset, validation_dataset, test_dataset